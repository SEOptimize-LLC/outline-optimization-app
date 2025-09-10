# Blog Outline Optimizer 🤖

An AI-powered Streamlit app that optimizes blog outlines for better search intent alignment while preserving all original content.

## Features

- **100% Content Preservation**: Never loses any original talking points
- **AI-Enhanced Analysis**: Uses OpenAI or Google Gemini for smart optimization
- **Search Intent Detection**: Automatically classifies content intent
- **Fan-Out Integration**: Incorporates keywords, user questions, and gaps
- **Clean Markdown Output**: Ready-to-use formatted content
- **Fallback Mode**: Works without AI using rule-based optimization

## Quick Start

### Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"** and connect your GitHub

4. **Configure deployment:**
   - Repository: `your-username/outline-optimizer`
   - Branch: `main`
   - Main file: `app.py`

5. **Add API keys (optional)** in App Settings → Secrets:
   ```toml
   OPENAI_API_KEY = "sk-..."
   GEMINI_API_KEY = "AIza..."
   ```

6. **Click Deploy!**

### Local Development

```bash
# Clone the repo
git clone https://github.com/your-username/outline-optimizer.git
cd outline-optimizer

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## How to Use

1. **Prepare your files:**
   - **Outline Draft** (.md): Your blog outline with headings and bullet points
   - **Fan-Out Analysis** (.md): Keywords, user questions, and gaps

2. **Upload both files** to the app

3. **Select optimization mode:**
   - AI-powered (if API keys configured)
   - Rule-based (always available)

4. **Click "Optimize Outline"**

5. **Download your enhanced outline**

## Input File Formats

### Outline Draft Example
```markdown
# Blog Title

## Section 1
- Talking point 1
- Talking point 2

### Subsection 1.1
- Detail point 1
- Detail point 2

## Section 2
- Talking point 1
```

### Fan-Out Analysis Example
```markdown
## Primary Keywords
- main keyword 1
- main keyword 2

## Related Keywords
- related term 1
- related term 2

## User Questions
- How to...?
- What is...?

## Competitive Gaps
- Missing topic 1
- Missing topic 2

## Content Themes
- Theme 1
- Theme 2
```

## API Configuration

The app supports two AI providers (both optional):

### OpenAI
- Models: o1-mini, o3, o3-pro (reasoning models)
- Add key: `OPENAI_API_KEY` in Streamlit secrets
- Note: o3 and o3-pro models might require special access

### Google Gemini
- Models: Gemini 2.0 Flash (experimental), Gemini 1.5 Pro, Gemini 1.5 Flash
- Add key: `GEMINI_API_KEY` in Streamlit secrets
- Note: Using latest available Gemini models

**Note:** The app works without API keys using rule-based optimization.

## File Structure

```
outline-optimizer/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
└── .streamlit/
    └── config.toml    # UI configuration (optional)
```

## Optimization Process

1. **Search Intent Analysis**: Classifies content intent
2. **Cross-Reference Analysis**: Maps fan-out data to outline
3. **Content Enhancement**: Adds relevant points while preserving original
4. **Quality Validation**: Ensures all guidelines are met
5. **Markdown Generation**: Creates production-ready output

## Key Principles

- ✅ Never removes original content
- ✅ Adds value through strategic enhancements
- ✅ Maintains logical structure and flow
- ✅ Integrates keywords naturally
- ✅ Delivers writer-ready markdown

## Support

For issues or questions, please open an issue on GitHub.

## License

MIT License - free to use and modify

---

Built with ❤️ using Streamlit
