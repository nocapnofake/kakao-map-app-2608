import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# 1. 환경변수 및 초기 설정
# ---------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="주소 근처 장소 찾기 에이전트",
    page_icon="📍",
    layout="wide",
)

# 세션 상태 초기화 (검색 기록 누적)
if "history" not in st.session_state:
    st.session_state.history = []

# 카테고리 매핑 딕셔너리
CATEGORY_MAP = {
    "음식점": "FD6",
    "카페": "CE7",
    "편의점": "CS2",
    "주차장": "PK6",
}


# ---------------------------------------------------------
# 2. 핵심 로직 함수 (기존 로직 재사용)
# ---------------------------------------------------------
def geocode_address(address: str):
    """카카오 주소 검색 API를 통해 주소를 (x, y) 좌표로 변환"""
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        documents = data.get("documents", [])
        if not documents:
            return None

        # x: 경도(lon), y: 위도(lat)
        return documents[0]["x"], documents[0]["y"]
    except requests.exceptions.RequestException as e:
        st.error(f"주소 검색 API 통신 오류: {e}")
        return None


def search_nearby_places(
    x: str, y: str, category_group_code: str, radius: int = 1000
):
    """카카오 카테고리 검색 API를 통해 좌표 기준 반경 내 장소를 거리순으로 검색"""
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {
        "category_group_code": category_group_code,
        "x": x,
        "y": y,
        "radius": radius,
        "sort": "distance",
        "size": 15,
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("documents", [])
    except requests.exceptions.RequestException as e:
        st.error(f"카테고리 검색 API 통신 오류: {e}")
        return []


def format_places_data(places: list) -> str:
    """검색 결과 리스트를 LLM 프롬프트에 전달할 텍스트 형태로 포맷팅"""
    if not places:
        return ""

    formatted = []
    for idx, place in enumerate(places, 1):
        name = place.get("place_name", "정보 없음")
        road_address = place.get("road_address_name") or place.get(
            "address_name", "주소 없음"
        )
        distance = place.get("distance", "0")
        phone = place.get("phone", "전화번호 없음")

        formatted.append(
            f"{idx}. 상호명: {name}\n"
            f"   - 주소: {road_address}\n"
            f"   - 거리: {distance}m\n"
            f"   - 연락처: {phone}"
        )

    return "\n".join(formatted)


def summarize_places_with_llm(
    client: OpenAI, address: str, category_name: str, places_text: str
) -> str:
    """LLM을 통해 검색된 장소 목록을 주관적 평가 없이 거리/도보시간 위주로 정리"""
    system_instruction = (
        "너는 정확하고 객관적인 '위치 정보 안내 비서'야.\n"
        "제공된 장소 목록 데이터만을 바탕으로 사용자에게 명확하고 읽기 쉽게 정리해줘.\n\n"
        "[지침]\n"
        "1. 맛, 분위기, 서비스, 인기 등 주관적이거나 검증되지 않은 '추천/평가'는 절대 하지 않는다.\n"
        "2. 거리(미터)를 기준으로 가까운 순서대로 일목요연하게 정리한다.\n"
        "3. 성인 평균 도보 속도(약 65m/분, 4km/h)를 기준으로 각 장소까지의 예상 도보 이동 시간을 계산해 명시한다.\n"
        "4. 군더더기 없는 정중하고 깔끔한 말투를 사용한다."
    )

    user_prompt = (
        f"기준 주소: {address}\n"
        f"검색 카테고리: {category_name}\n\n"
        f"[검색 결과 목록]\n"
        f"{places_text}\n\n"
        f"위 데이터를 바탕으로 각 장소의 상호명, 주소, 거리 및 예상 도보 이동 시간을 깔끔하게 정리해줘."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# 3. Streamlit UI 레이아웃
# ---------------------------------------------------------
st.title("📍 카카오맵 주소 근처 장소 찾기 에이전트")
st.caption(
    "주소와 카테고리를 설정하면 반경 내 장소를 검색하고 AI가 도보 이동 정보와 함께 요약해 드립니다."
)

# API 키 누락 검사
if not OPENAI_API_KEY or not KAKAO_API_KEY:
    st.error(
        "⚠️ .env 파일에 `OPENAI_API_KEY` 또는 `KAKAO_API_KEY`가 설정되어 있지 않습니다."
    )
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 사이드바 입력 위젯
with st.sidebar:
    st.header("🔍 검색 조건 설정")
    address_input = st.text_input(
        "기준 주소 입력", placeholder="예: 서울특별시 종로구 세종대로 209"
    )
    selected_category = st.selectbox(
        "카테고리 선택", options=list(CATEGORY_MAP.keys())
    )
    radius_input = st.slider(
        "검색 반경 (미터)",
        min_value=100,
        max_value=3000,
        value=1000,
        step=100,
    )
    search_button = st.button(
        "장소 검색하기", type="primary", use_container_width=True
    )

# 검색 버튼 클릭 시 동작
if search_button:
    if not address_input.strip():
        st.error("주소를 입력해 주세요.")
    else:
        with st.spinner("주소 좌표 변환 및 주변 장소 검색 중..."):
            # 1. 주소 -> 좌표 변환
            coords = geocode_address(address_input.strip())

            if not coords:
                st.error(
                    "입력하신 주소를 찾을 수 없습니다. 도로명 또는 지번 주소를 정확히 입력해주세요."
                )
            else:
                lon_x, lat_y = coords
                category_code = CATEGORY_MAP[selected_category]

                # 2. 카테고리 검색
                places = search_nearby_places(
                    x=lon_x,
                    y=lat_y,
                    category_group_code=category_code,
                    radius=radius_input,
                )

                if not places:
                    st.error(
                        f"반경 {radius_input}m 이내에 검색된 '{selected_category}' 결과가 없습니다."
                    )
                else:
                    # 3. 데이터프레임 구성 (지도 표시용 lat, lon 포함)
                    table_rows = []
                    map_rows = []

                    for place in places:
                        name = place.get("place_name", "정보 없음")
                        addr = place.get("road_address_name") or place.get(
                            "address_name", "주소 없음"
                        )
                        dist = int(place.get("distance", 0))
                        p_lat = float(place.get("y"))
                        p_lon = float(place.get("x"))

                        table_rows.append(
                            {
                                "상호명": name,
                                "주소": addr,
                                "거리(m)": dist,
                                "전화번호": place.get("phone", "-"),
                            }
                        )
                        map_rows.append({"lat": p_lat, "lon": p_lon})

                    df_places = pd.DataFrame(table_rows)
                    df_map = pd.DataFrame(map_rows)

                    # 4. LLM 요약 생성
                    places_text = format_places_data(places)
                    llm_summary = summarize_places_with_llm(
                        openai_client,
                        address_input,
                        selected_category,
                        places_text,
                    )

                    # 세션 상태에 기록 저장 (최신순 조회를 위해 맨 앞에 추가)
                    st.session_state.history.insert(
                        0,
                        {
                            "address": address_input,
                            "category": selected_category,
                            "radius": radius_input,
                            "count": len(places),
                            "summary": llm_summary,
                            "df": df_places,
                            "map_df": df_map,
                        },
                    )

# 결과 화면 렌더링
if st.session_state.history:
    latest = st.session_state.history[0]

    st.subheader(
        f"📌 검색 결과: {latest['address']} ({latest['category']} / 반경 {latest['radius']}m)"
    )

    # 1. LLM 안내/요약 표시
    st.info(latest["summary"])

    # 2. 지도 및 데이터프레임 나란히 배치
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🗺️ 위치 지도")
        st.map(latest["map_df"], zoom=14)

    with col2:
        st.markdown("#### 📋 장소 상세 목록")
        st.dataframe(latest["df"], use_container_width=True, hide_index=True)

    # 3. 지난 검색 기록 (누적 히스토리)
    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("🕒 지난 검색 기록")

        for idx, item in enumerate(st.session_state.history[1:], start=1):
            with st.expander(
                f"[{idx}] {item['address']} - {item['category']} ({item['count']}개 검색됨)"
            ):
                st.write(f"**반경:** {item['radius']}m")
                st.markdown("**AI 요약 내용:**")
                st.write(item["summary"])
                st.dataframe(
                    item["df"], use_container_width=True, hide_index=True
                )