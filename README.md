# K2 Learn — Interview & Career Suite

A Streamlit-based, AI-powered platform for interview preparation and study, leveraging K2 Think (simulated via GPT-2) with RAG, live interviews, MCQs, flashcards, a code sandbox, and a resume builder with job matching. Built for the K2 Think Hackathon (October 2025).

**Live Demo**: [https://k2-learn.streamlit.app](https://k2-learn.streamlit.app)  
**Demo Video**: [Google Drive](https://drive.google.com/file/d/1dEZqkuYhTFigbQflE33i1JA4q331Iftn/view?usp=sharing)



https://github.com/user-attachments/assets/50b2dcda-a584-43fb-9d3b-8263c60c604d



## Features

- **Study Hub**: Upload PDF/TXT/images to build a knowledge base (RAG) using FAISS and SentenceTransformer (`all-MiniLM-L6-v2`).
- **K2 Chat (RAG)**: Contextual Q&A with user-uploaded notes, powered by GPT-2 (placeholder for K2 Think API).
- **MCQ Generator**: Creates multiple-choice questions from uploaded notes or inline text, with immediate feedback.
- **Flashcards**: Generates flippable flashcards (Aptitude/Technical/Coding) with JavaScript-based UI.
- **Interview Simulator**: Supports text/audio interviews (Technical/Behavioral/Visa) with scoring, using SpeechRecognition (Google Web Speech API) and pyttsx3 for TTS.
- **Code Sandbox**: Safe Python execution with AI code review via GPT-2.
- **Resume Builder & Job Matcher**: Generates ATS-optimized resumes and matches jobs using embeddings or keyword overlap.
- **Export Logs**: Encrypts interview logs with `cryptography.Fernet` for privacy.
- **Avatar Integration**: ReadyPlayerMe iframe with mood mapping based on interview scores.
- **Visualizations**: Plotly charts for interview scores and job match rankings.

## Tech Stack

- **Frontend**: Streamlit, streamlit-ace, streamlit-webrtc, Plotly
- **AI Models**: GPT-2 (`transformers`), SentenceTransformer (`all-MiniLM-L6-v2`), spaCy (`en_core_web_sm`)
- **Backend**: Python, FAISS (RAG), PyPDF2 (PDF processing), pytesseract (OCR), SpeechRecognition, pyttsx3 (TTS)
- **Security**: cryptography (Fernet for log encryption)
- **Fallbacks**: NumPy for similarity (if FAISS fails), audio uploads (if WebRTC unavailable)

## Prerequisites

- Python 3.8+
- Tesseract OCR installed system-wide ([Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html))
- Optional: `.env` file with `ELEVENLABS_API_KEY` for enhanced TTS (falls back to pyttsx3)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/thinkspace/k2-think-demo.git
   cd k2-think-demo
   ```

2. **Install Dependencies**:
   ```bash
   pip install streamlit transformers sentence-transformers faiss-cpu pypdf2 pytesseract pillow speechrecognition pyttsx3 streamlit-webrtc spacy streamlit-ace cryptography plotly
   python -m spacy download en_core_web_sm
   ```

3. **(Optional) Set Up .env**:
   ```bash
   echo "ELEVENLABS_API_KEY=your_key" > .env
   ```

4. **Install Tesseract**:
   - Ubuntu: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

1. **Run the App**:
   ```bash
   streamlit run k2_think_full_app.py
   ```

2. **Navigate Pages**:
   - **Home**: Overview and metrics (KB chunks, flashcards, MCQs).
   - **Study Hub**: Upload PDF/TXT/images to build RAG knowledge base.
   - **K2 Chat**: Ask questions with RAG context (top-k snippets).
   - **MCQ/Flashcards**: Generate and practice from KB or inline text.
   - **Interview (Live)**: Simulate text/audio interviews with scoring and avatar feedback.
   - **Code Sandbox**: Write/run Python code with AI review.
   - **Resume Builder**: Create resumes and match jobs.
   - **Export Logs**: Download encrypted interview logs.

3. **Example Workflow**:
   - Upload a PDF of tech notes in Study Hub.
   - Ask, “Explain binary search” in K2 Chat (RAG retrieves relevant snippets).
   - Generate 5 MCQs or flashcards for practice.
   - Run a technical interview with audio responses (upload WAV if WebRTC fails).
   - Build a resume and match to sample jobs.
   - Export logs with a password.

## Troubleshooting

- **RAG Fails**: Ensure FAISS and SentenceTransformers are installed; check logs for errors.
- **Audio Issues**: WebRTC may require browser permissions; use audio upload fallback.
- **TTS Errors**: Install pyttsx3 or provide ElevenLabs API key in `.env`.
- **PDF/Image Uploads**: Verify PyPDF2 and pytesseract installations.
- **Console Logs**: Check terminal for detailed error messages (`logging` module).

## Limitations

- **K2 Think API**: Simulated with GPT-2 due to unavailable K2 Think endpoints; replace with `/chat/completions` when available.
- **Live Audio**: Streamlit’s WebRTC is unstable; audio uploads used as fallback.
- **Job Matching**: Uses mock job data; integrate Indeed API for real listings.
- **Performance**: GPT-2 is lightweight but less coherent than larger models; fine-tuning or cloud hosting recommended.

## Contributing

1. Fork the repo and create a branch (`git checkout -b feature-name`).
2. Commit changes (`git commit -m "Add feature X"`).
3. Push to your fork (`git push origin feature-name`).
4. Open a pull request with a clear description.


## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built for the K2 Think Hackathon, October 2025.
- Powered by Hugging Face, Streamlit, and open-source libraries.
- Inspired by the need for accessible career and study tools.

---

### Notes
- **Alignment with One-Pager**: The README reflects the one-pager’s features, tech stack (corrected to Streamlit), and demo links, ensuring consistency.
- **Ease of Use**: Clear setup and usage instructions cater to hackathon judges and developers.
- **Placeholders**: GitHub repo and demo links are from the one-pager; update with actual URLs.
- **File Creation**: Save as `README.md` in the repo root. Use Markdown viewer or GitHub for formatting.
- **Customization**: Replace the logo URL and team contact with real details if available.

If you need help setting up the repo, deploying to Streamlit Cloud, or adding specific sections (e.g., API integration guide), let me know!
