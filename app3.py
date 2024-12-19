import streamlit as st
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# KoBART 모델 및 토크나이저 로드
model_name = 'gogamza/kobart-summarization'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 뉴스 요약 함수
def summarize_article(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=130,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit 애플리케이션 설정
st.title("뉴스 요약 웹사이트")
st.write("텍스트를 입력하면 간결한 요약문을 생성해줍니다.")

# 사용자 입력
user_input = st.text_area(
    "요약할 뉴스 기사 본문을 입력하세요:",
    height=300,
    placeholder="여기에 요약하고 싶은 뉴스 기사를 입력해주세요..."
)

# 요약 버튼
if st.button("요약 실행"):
    if user_input.strip():
        with st.spinner("요약 중입니다... 잠시만 기다려주세요."):
            try:
                summary = summarize_article(user_input)
                st.subheader("요약 결과")
                st.write(summary)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
    else:
        st.error("유효한 텍스트를 입력하세요.")
