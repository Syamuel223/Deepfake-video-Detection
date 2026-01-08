# app.py
import streamlit as st
import tempfile
from collections import Counter
from scripts.predicts import load_trained_model, predict_video

def main():
    st.set_page_config(page_title="Deepfake Video Detector", layout="centered")
    st.title("🎭 Deepfake Video Detection App")
    st.write("Upload a short video clip (mp4) and detect if it's Real or Fake.")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.video(uploaded_file)

        with st.spinner("Analyzing video..."):
            model = load_trained_model()
            predictions = predict_video(video_path, model)

        st.subheader("🧠 Frame-by-Frame Predictions:")
        frame_results = []
        for i, pred in enumerate(predictions):
            label = "No face detected"
            if pred is not None:
                label = "Fake" if pred == 1 else "Real"
            frame_results.append(f"Frame {i+1}: {label}")
        st.write("\n".join(frame_results))

        valid_preds = [p for p in predictions if p is not None]
        if valid_preds:
            decision = Counter(valid_preds)
            result = "Fake" if decision[1] > decision[0] else "Real"
            st.success(f"🎯 Final Verdict: **{result.upper()}** ({decision[0]} Real, {decision[1]} Fake)")
        else:
            st.warning("⚠️ No faces detected in any frame. Unable to classify.")

if __name__ == "__main__":
    main()
