import streamlit as st
import pandas as pd
import re
from datetime import datetime
import json
import os
from io import StringIO

# Try to import AI libraries (optional)
try:
    from openai import OpenAI  # FIXED: Using new OpenAI client
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI-Powered Blog Outline Optimizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3d59;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .step-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metrics-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e3d59;
    }
    .ai-insight {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4169e1;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
    .api-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .api-available {
        background-color: #d4edda;
        color: #155724;
    }
    .api-unavailable {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'original_outline' not in st.session_state:
    st.session_state.original_outline = None
if 'fanout_data' not in st.session_state:
    st.session_state.fanout_data = None
if 'optimized_outline' not in st.session_state:
    st.session_state.optimized_outline = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

class AIConfigManager:
    """Manages API configurations and auto-detection"""
    
    def __init__(self):
        self.available_apis = {}
        self.check_available_apis()
        
    def check_available_apis(self):
        """Auto-detect which APIs are configured in Streamlit secrets"""
        
        # Check for OpenAI
        if OPENAI_AVAILABLE:
            try:
                if 'OPENAI_API_KEY' in st.secrets:
                    # FIXED: Create OpenAI client with new API
                    client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
                    self.available_apis['openai'] = {
                        'configured': True,
                        'models': ['gpt-4.1-mini', 'gpt-5', 'gpt-4o-mini'],  # YOUR models preserved
                        'default_model': 'gpt-4-turbo-preview',
                        'display_name': 'OpenAI',
                        'api_key': st.secrets['OPENAI_API_KEY'],
                        'client': client  # Store the client
                    }
                else:
                    self.available_apis['openai'] = {'configured': False}
            except:
                self.available_apis['openai'] = {'configured': False}
        
        # Check for Google Gemini
        if GEMINI_AVAILABLE:
            try:
                if 'GEMINI_API_KEY' in st.secrets:
                    genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
                    self.available_apis['gemini'] = {
                        'configured': True,
                        'models': ['gemini-2.5-pro', 'gemini-2.5-flash'],  # YOUR models preserved
                        'default_model': 'gemini-1.5-pro',
                        'display_name': 'Google Gemini',
                        'api_key': st.secrets['GEMINI_API_KEY']
                    }
                else:
                    self.available_apis['gemini'] = {'configured': False}
            except:
                self.available_apis['gemini'] = {'configured': False}
    
    def get_available_providers(self):
        """Return list of available AI providers"""
        available = []
        for provider, config in self.available_apis.items():
            if config.get('configured', False):
                available.append({
                    'id': provider,
                    'name': config['display_name'],
                    'models': config['models']
                })
        return available
    
    def has_any_api(self):
        """Check if any API is configured"""
        return any(api.get('configured', False) for api in self.available_apis.values())

class OutlineOptimizer:
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.ai_provider = None
        self.model = None
        self.ai_enabled = False
        self.openai_client = None  # Store OpenAI client
        self.intent_types = {
            'informational': 'Knowledge-seeking, how-to guides, explanations',
            'navigational': 'Specific websites, brands, resources',
            'commercial': 'Product/service research before purchase',
            'transactional': 'Ready to take action (buy, subscribe, download)',
            'local': 'Location-based information or services'
        }
    
    def set_provider(self, provider_id, model=None):
        """Set the AI provider and model to use"""
        if self.config_manager and provider_id in self.config_manager.available_apis:
            config = self.config_manager.available_apis[provider_id]
            if config.get('configured', False):
                self.ai_provider = provider_id
                self.model = model or config['default_model']
                self.ai_enabled = True
                # FIXED: Get the OpenAI client if available
                if provider_id == 'openai' and 'client' in config:
                    self.openai_client = config['client']
                return True
        return False
    
    def parse_outline(self, outline_text):
        """Parse the original outline structure"""
        lines = outline_text.strip().split('\n')
        structure = {
            'title': '',
            'sections': [],
            'total_bullets': 0,
            'heading_count': {'h1': 0, 'h2': 0, 'h3': 0}
        }
        
        current_section = None
        current_subsection = None
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Count bullets
            if line_stripped.startswith('- ') or line_stripped.startswith('* '):
                structure['total_bullets'] += 1
                bullet_content = line_stripped[2:]
                if current_subsection:
                    current_subsection['bullets'].append(bullet_content)
                elif current_section:
                    current_section['bullets'].append(bullet_content)
            
            # Parse headings
            elif line_stripped.startswith('# '):
                structure['title'] = line_stripped[2:]
                structure['heading_count']['h1'] += 1
                
            elif line_stripped.startswith('## '):
                if current_section:
                    structure['sections'].append(current_section)
                current_section = {
                    'title': line_stripped[3:],
                    'bullets': [],
                    'subsections': []
                }
                current_subsection = None
                structure['heading_count']['h2'] += 1
                
            elif line_stripped.startswith('### '):
                if current_section:
                    current_subsection = {
                        'title': line_stripped[4:],
                        'bullets': []
                    }
                    current_section['subsections'].append(current_subsection)
                structure['heading_count']['h3'] += 1
        
        # Add last section
        if current_section:
            structure['sections'].append(current_section)
            
        return structure
    
    def parse_fanout(self, fanout_text):
        """Parse the fan-out analysis report"""
        data = {
            'primary_keywords': [],
            'related_keywords': [],
            'user_questions': [],
            'search_volumes': {},
            'competitive_gaps': [],
            'content_themes': []
        }
        
        lines = fanout_text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Identify sections by headers
            line_lower = line.lower()
            if 'primary keyword' in line_lower:
                current_section = 'primary'
            elif 'related keyword' in line_lower or 'secondary' in line_lower:
                current_section = 'related'
            elif 'question' in line_lower and 'user' in line_lower:
                current_section = 'questions'
            elif 'gap' in line_lower and 'competitive' in line_lower:
                current_section = 'gaps'
            elif 'theme' in line_lower or 'topic' in line_lower:
                current_section = 'themes'
            elif 'search volume' in line_lower:
                current_section = 'volumes'
            elif line.startswith('- ') or line.startswith('* '):
                content = line[2:].strip()
                if current_section == 'primary':
                    data['primary_keywords'].append(content)
                elif current_section == 'related':
                    data['related_keywords'].append(content)
                elif current_section == 'questions':
                    data['user_questions'].append(content)
                elif current_section == 'gaps':
                    data['competitive_gaps'].append(content)
                elif current_section == 'themes':
                    data['content_themes'].append(content)
                elif current_section == 'volumes':
                    # Parse search volume if it contains numbers
                    parts = content.split('-')
                    if len(parts) >= 2:
                        keyword = parts[0].strip().strip('"')
                        volume = parts[1].strip()
                        data['search_volumes'][keyword] = volume
        
        return data
    
    def analyze_with_ai(self, outline_text, fanout_text):
        """Comprehensive AI analysis of outline and fan-out data"""
        
        if not self.ai_enabled:
            return self.fallback_analysis(outline_text, fanout_text)
        
        prompt = self._create_analysis_prompt(outline_text, fanout_text)
        
        if self.ai_provider == 'openai':
            return self._openai_analysis(prompt)
        elif self.ai_provider == 'gemini':
            return self._gemini_analysis(prompt)
        else:
            return self.fallback_analysis(outline_text, fanout_text)
    
    def _create_analysis_prompt(self, outline_text, fanout_text):
        """Create comprehensive prompt for AI analysis"""
        return f"""
        You are an expert SEO and content strategist following specific optimization guidelines.
        
        ORIGINAL OUTLINE:
        {outline_text[:3000]}
        
        FAN-OUT ANALYSIS (Keywords and User Research):
        {fanout_text[:2000]}
        
        OPTIMIZATION GUIDELINES:
        
        1. SEARCH INTENT ANALYSIS:
        - Classify intent: informational, navigational, commercial, transactional, or local
        - Determine user goals and content depth
        - Identify optimal content type
        
        2. CROSS-REFERENCE ANALYSIS:
        - Evaluate outline coherence
        - Identify content gaps from fan-out
        - Map fan-out insights to sections
        - NEVER suggest removing content
        
        3. OUTLINE OPTIMIZATION:
        - Optimize headings for natural language
        - PRESERVE all original talking points
        - Add complementary points from fan-out
        - Structure for user journey
        
        4. TALKING POINT RULES:
        - Maintain EVERY original point
        - Add supporting details
        - Include user questions
        - Integrate keywords naturally
        
        Provide JSON response with exactly these fields:
        {{
            "search_intent": {{
                "type": "informational|commercial|transactional|navigational",
                "confidence": 0.0 to 1.0,
                "reasoning": "brief explanation"
            }},
            "content_gaps": ["gap1", "gap2", "gap3"],
            "enhancement_suggestions": {{
                "section_name": ["suggestion1", "suggestion2"]
            }},
            "keyword_mapping": {{
                "section_name": ["keyword1", "keyword2"]
            }},
            "user_questions_mapping": {{
                "section_name": ["question1", "question2"]
            }},
            "quality_score": 0-100,
            "priority_improvements": ["improvement1", "improvement2", "improvement3"],
            "content_type": "guide|how-to|listicle|comparison|review",
            "estimated_read_time": "X-Y minutes",
            "competitive_advantages": ["advantage1", "advantage2"]
        }}
        
        Return ONLY valid JSON, no markdown formatting.
        """
    
    def _openai_analysis(self, prompt):
        """Process with OpenAI using NEW client API"""
        if not self.openai_client:
            st.error("OpenAI client not initialized")
            return self.fallback_analysis("", "")
        
        try:
            # Models that need max_completion_tokens instead of max_tokens
            new_models = ['gpt-5', 'gpt-4.1-mini', 'gpt-4o-mini', 'o1-mini', 'o1-preview']
            
            # FIXED: Using correct parameter based on model
            if self.model in new_models:
                # Use max_completion_tokens for newer models
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert SEO analyst. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_completion_tokens=2000  # FIXED: Using max_completion_tokens
                )
            else:
                # Use max_tokens for older models (fallback)
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert SEO analyst. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
            
            response_text = response.choices[0].message.content
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Clean JSON extraction
            if '{' in response_text:
                response_text = response_text[response_text.index('{'):]
            if '}' in response_text:
                response_text = response_text[:response_text.rindex('}')+1]
                
            result = json.loads(response_text)
            return result
            
        except Exception as e:
            # If max_completion_tokens fails, try with max_tokens as fallback
            if "max_tokens" in str(e) or "max_completion_tokens" in str(e):
                try:
                    st.warning("Adjusting token parameter for model compatibility...")
                    # Try opposite parameter
                    if "max_tokens" in str(e):
                        response = self.openai_client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": "You are an expert SEO analyst. Return only valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_completion_tokens=2000
                        )
                    else:
                        response = self.openai_client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": "You are an expert SEO analyst. Return only valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_tokens=2000
                        )
                    
                    response_text = response.choices[0].message.content
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                    
                    if '{' in response_text:
                        response_text = response_text[response_text.index('{'):]
                    if '}' in response_text:
                        response_text = response_text[:response_text.rindex('}')+1]
                        
                    result = json.loads(response_text)
                    return result
                except Exception as retry_error:
                    st.error(f"AI Analysis Error: {str(retry_error)}")
                    return self.fallback_analysis("", "")
            else:
                st.error(f"AI Analysis Error: {str(e)}")
                return self.fallback_analysis("", "")
    
    def _gemini_analysis(self, prompt):
        """Process with Google Gemini"""
        try:
            model = genai.GenerativeModel(self.model)
            
            response = model.generate_content(
                prompt + "\n\nIMPORTANT: Return ONLY the JSON object.",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2000,
                )
            )
            
            response_text = response.text
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Clean JSON extraction
            if '{' in response_text:
                response_text = response_text[response_text.index('{'):]
            if '}' in response_text:
                response_text = response_text[:response_text.rindex('}')+1]
                
            result = json.loads(response_text)
            return result
            
        except Exception as e:
            st.error(f"AI Analysis Error: {str(e)}")
            return self.fallback_analysis("", "")
    
    def analyze_search_intent(self, outline_structure, fanout_data):
        """Determine search intent from keywords and structure"""
        intent_signals = {
            'informational': 0,
            'commercial': 0,
            'transactional': 0,
            'navigational': 0,
            'local': 0
        }
        
        # Analyze keywords for intent signals
        informational_terms = ['how', 'what', 'why', 'guide', 'tutorial', 'learn', 'understand']
        commercial_terms = ['best', 'review', 'compare', 'top', 'vs', 'comparison']
        transactional_terms = ['buy', 'purchase', 'price', 'deal', 'discount', 'order']
        
        all_keywords = ' '.join(
            fanout_data.get('primary_keywords', []) + 
            fanout_data.get('related_keywords', [])
        ).lower()
        
        for term in informational_terms:
            if term in all_keywords:
                intent_signals['informational'] += 2
        
        for term in commercial_terms:
            if term in all_keywords:
                intent_signals['commercial'] += 2
                
        for term in transactional_terms:
            if term in all_keywords:
                intent_signals['transactional'] += 2
        
        # Determine primary intent
        primary_intent = max(intent_signals, key=intent_signals.get)
        
        return {
            'primary_intent': primary_intent,
            'intent_scores': intent_signals,
            'content_type': self.determine_content_type(primary_intent),
            'user_goal': self.determine_user_goal(primary_intent, fanout_data)
        }
    
    def determine_content_type(self, intent):
        """Map intent to content type"""
        content_map = {
            'informational': 'Comprehensive Guide',
            'commercial': 'Comparison/Review',
            'transactional': 'Product Page',
            'navigational': 'Resource Hub',
            'local': 'Location Guide'
        }
        return content_map.get(intent, 'General Article')
    
    def determine_user_goal(self, intent, fanout_data):
        """Determine specific user goal"""
        questions = fanout_data.get('user_questions', [])
        if questions:
            return f"Answer: {questions[0]}"
        return "Provide comprehensive information on the topic"
    
    def calculate_relevance(self, text1, text2):
        """Calculate relevance score between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def generate_enhancements(self, section_title, fanout_data):
        """Generate enhancement bullets based on fanout data"""
        enhancements = []
        section_lower = section_title.lower()
        
        # Add relevant questions
        for question in fanout_data.get('user_questions', []):
            relevance = self.calculate_relevance(section_lower, question.lower())
            if relevance > 0.3:
                enhancements.append(f"Answer: {question}")
                if len(enhancements) >= 2:
                    break
        
        # Add competitive gaps
        for gap in fanout_data.get('competitive_gaps', []):
            relevance = self.calculate_relevance(section_lower, gap.lower())
            if relevance > 0.3 and len(enhancements) < 3:
                enhancements.append(f"Address gap: {gap}")
        
        # Add content themes
        for theme in fanout_data.get('content_themes', []):
            if any(word in section_lower for word in theme.lower().split()) and len(enhancements) < 4:
                enhancements.append(f"Include theme: {theme}")
        
        # Add relevant keywords
        relevant_keywords = []
        for kw in fanout_data.get('related_keywords', []):
            if any(word in section_lower for word in kw.lower().split()):
                relevant_keywords.append(kw)
        
        if relevant_keywords and len(enhancements) < 5:
            enhancements.append(f"Keywords to incorporate: {', '.join(relevant_keywords[:3])}")
        
        return enhancements
    
    def optimize_outline(self, original_structure, fanout_data, intent_analysis, ai_analysis=None):
        """Optimize the outline while preserving all original content"""
        optimized = {
            'title': self.optimize_title(original_structure['title'], fanout_data),
            'sections': [],
            'metadata': {
                'original_bullets': original_structure['total_bullets'],
                'new_bullets': 0,
                'expansion_rate': 0,
                'primary_intent': intent_analysis['primary_intent']
            }
        }
        
        # Process each section
        for section in original_structure['sections']:
            opt_section = {
                'title': self.optimize_heading(section['title'], fanout_data),
                'bullets': section['bullets'].copy(),  # Preserve all original bullets
                'subsections': []
            }
            
            # Add AI or rule-based enhancements
            if ai_analysis and 'enhancement_suggestions' in ai_analysis:
                section_suggestions = ai_analysis['enhancement_suggestions'].get(section['title'], [])
                opt_section['bullets'].extend(section_suggestions[:3])
            else:
                enhancements = self.generate_enhancements(section['title'], fanout_data)
                opt_section['bullets'].extend(enhancements)
            
            # Process subsections
            for subsection in section.get('subsections', []):
                opt_subsection = {
                    'title': self.optimize_heading(subsection['title'], fanout_data),
                    'bullets': subsection['bullets'].copy()  # Preserve all original bullets
                }
                # Add enhancements
                sub_enhancements = self.generate_enhancements(subsection['title'], fanout_data)
                opt_subsection['bullets'].extend(sub_enhancements[:2])
                opt_section['subsections'].append(opt_subsection)
            
            optimized['sections'].append(opt_section)
        
        # Add FAQ section if questions exist
        if fanout_data.get('user_questions'):
            faq_section = self.create_faq_section(fanout_data['user_questions'])
            optimized['sections'].append(faq_section)
        
        # Calculate metrics
        new_bullet_count = sum(
            len(s['bullets']) + sum(len(ss['bullets']) for ss in s['subsections'])
            for s in optimized['sections']
        )
        optimized['metadata']['new_bullets'] = new_bullet_count
        optimized['metadata']['expansion_rate'] = (
            ((new_bullet_count - original_structure['total_bullets']) / 
             original_structure['total_bullets'] * 100) if original_structure['total_bullets'] > 0 else 0
        )
        
        return optimized
    
    def optimize_title(self, original_title, fanout_data):
        """Optimize the main title for search"""
        primary_kw = fanout_data.get('primary_keywords', [])
        if primary_kw and primary_kw[0].lower() not in original_title.lower():
            # Add primary keyword if not present
            return f"{original_title}: {primary_kw[0].title()}"
        return original_title
    
    def optimize_heading(self, heading, fanout_data):
        """Optimize section headings"""
        # Simple optimization - maintains readability
        return heading
    
    def create_faq_section(self, questions):
        """Create FAQ section from user questions"""
        return {
            'title': 'Frequently Asked Questions',
            'bullets': [f"Q: {q}" for q in questions[:5]],
            'subsections': []
        }
    
    def generate_markdown(self, optimized_outline):
        """Generate clean markdown output"""
        md = f"# {optimized_outline['title']}\n\n"
        
        for section in optimized_outline['sections']:
            md += f"## {section['title']}\n"
            for bullet in section['bullets']:
                md += f"- {bullet}\n"
            md += "\n"
            
            for subsection in section['subsections']:
                md += f"### {subsection['title']}\n"
                for bullet in subsection['bullets']:
                    md += f"- {bullet}\n"
                md += "\n"
        
        return md
    
    def fallback_analysis(self, outline_text, fanout_text):
        """Rule-based fallback analysis"""
        # Count sections and bullets
        section_count = len(re.findall(r'^##\s+', outline_text, re.MULTILINE))
        bullet_count = len(re.findall(r'^[-*]\s+', outline_text, re.MULTILINE))
        
        # Estimate read time
        estimated_words = bullet_count * 20 + section_count * 50
        read_time_min = int(estimated_words / 200)
        read_time_max = int(estimated_words / 150)
        
        return {
            "search_intent": {
                "type": "informational",
                "confidence": 0.7,
                "reasoning": "Rule-based analysis"
            },
            "content_gaps": [
                "Add specific examples and case studies",
                "Include data and statistics",
                "Add actionable tips"
            ],
            "enhancement_suggestions": {},
            "keyword_mapping": {},
            "user_questions_mapping": {},
            "quality_score": 75,
            "priority_improvements": [
                "Add FAQ section",
                "Enhance introduction",
                "Include examples"
            ],
            "content_type": "guide",
            "estimated_read_time": f"{read_time_min}-{read_time_max} minutes",
            "competitive_advantages": ["Comprehensive coverage"]
        }

def main():
    st.markdown('<h1 class="main-header">🤖 Blog Outline Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("Transform your blog outlines with AI-powered search intent optimization")
    
    # Initialize configuration
    config_manager = AIConfigManager() if (OPENAI_AVAILABLE or GEMINI_AVAILABLE) else None
    optimizer = OutlineOptimizer(config_manager)
    
    # Sidebar
    with st.sidebar:
        st.header("🔌 Configuration")
        
        # Show API status if AI libraries are available
        if config_manager:
            if config_manager.has_any_api():
                st.markdown('<div class="api-status api-available">✅ AI APIs Available</div>', 
                           unsafe_allow_html=True)
                
                # Provider selection
                available_providers = config_manager.get_available_providers()
                if available_providers:
                    st.divider()
                    
                    provider_options = ["Rule-based (No AI)"] + [p['name'] for p in available_providers]
                    selected_provider_name = st.selectbox(
                        "Select Optimization Mode",
                        provider_options,
                        help="Choose AI model or rule-based optimization"
                    )
                    
                    if selected_provider_name != "Rule-based (No AI)":
                        selected_provider = next(
                            (p for p in available_providers if p['name'] == selected_provider_name),
                            None
                        )
                        
                        if selected_provider:
                            selected_model = st.selectbox(
                                "Select Model",
                                selected_provider['models'],
                                help="Select your preferred model"
                            )
                            optimizer.set_provider(selected_provider['id'], selected_model)
                            st.success(f"Using: {selected_model}")
                    else:
                        st.info("Using rule-based optimization")
            else:
                st.markdown('<div class="api-status api-unavailable">❌ No APIs Configured</div>', 
                           unsafe_allow_html=True)
                st.warning("""
                Add API keys in Streamlit Secrets:
                - OPENAI_API_KEY
                - GEMINI_API_KEY
                """)
        else:
            st.info("AI libraries not installed. Using rule-based optimization.")
        
        st.divider()
        st.header("📋 How it Works")
        st.markdown("""
        1. Upload your outline draft (.md)
        2. Upload fan-out analysis (.md)
        3. Click optimize
        4. Download enhanced outline
        
        **Features:**
        - ✅ Preserves all original content
        - ✅ Adds search-optimized content
        - ✅ Ready-to-use markdown
        """)
        
        if st.button("🔄 Reset", type="secondary"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    # Main content
    st.markdown('<div class="step-header">📁 Step 1: Upload Files</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        outline_file = st.file_uploader(
            "Upload Blog Outline Draft",
            type=['md', 'txt'],
            help="Your original outline in Markdown"
        )
        if outline_file:
            st.session_state.original_outline = outline_file.read().decode('utf-8')
            st.success("✅ Outline uploaded!")
    
    with col2:
        fanout_file = st.file_uploader(
            "Upload Fan-Out Analysis",
            type=['md', 'txt'],
            help="Keywords and user research data"
        )
        if fanout_file:
            st.session_state.fanout_data = fanout_file.read().decode('utf-8')
            st.success("✅ Fan-out uploaded!")
    
    # Process if both files uploaded
    if st.session_state.original_outline and st.session_state.fanout_data:
        
        # Preview files
        with st.expander("📄 View Uploaded Files"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Outline:**")
                st.text(st.session_state.original_outline[:500] + "...")
            with col2:
                st.markdown("**Fan-Out Analysis:**")
                st.text(st.session_state.fanout_data[:500] + "...")
        
        st.markdown('<div class="step-header">⚙️ Step 2: Optimize</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Optimize Outline", type="primary", use_container_width=True):
            with st.spinner("Optimizing your outline..."):
                progress = st.progress(0)
                
                # Parse outline
                progress.progress(20)
                outline_structure = optimizer.parse_outline(st.session_state.original_outline)
                
                # Parse fan-out
                progress.progress(40)
                fanout_parsed = optimizer.parse_fanout(st.session_state.fanout_data)
                
                # Analyze intent
                progress.progress(60)
                intent_analysis = optimizer.analyze_search_intent(outline_structure, fanout_parsed)
                
                # AI analysis if enabled
                ai_analysis = None
                if optimizer.ai_enabled:
                    ai_analysis = optimizer.analyze_with_ai(
                        st.session_state.original_outline,
                        st.session_state.fanout_data
                    )
                    if 'search_intent' in ai_analysis:
                        intent_analysis.update(ai_analysis['search_intent'])
                
                st.session_state.analysis_results = intent_analysis
                
                # Optimize outline
                progress.progress(80)
                optimized = optimizer.optimize_outline(
                    outline_structure, 
                    fanout_parsed, 
                    intent_analysis,
                    ai_analysis
                )
                
                # Generate markdown
                progress.progress(100)
                st.session_state.optimized_outline = optimizer.generate_markdown(optimized)
                st.session_state.optimization_metadata = optimized['metadata']
                st.session_state.optimization_complete = True
                
                st.success("✅ Optimization complete!")
    
    # Display results
    if st.session_state.optimization_complete:
        st.markdown('<div class="step-header">📊 Step 3: Results</div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Original Bullets",
                st.session_state.optimization_metadata['original_bullets']
            )
        with col2:
            st.metric(
                "Enhanced Bullets",
                st.session_state.optimization_metadata['new_bullets'],
                delta=f"+{st.session_state.optimization_metadata['new_bullets'] - st.session_state.optimization_metadata['original_bullets']}"
            )
        with col3:
            st.metric(
                "Expansion Rate",
                f"{st.session_state.optimization_metadata['expansion_rate']:.1f}%"
            )
        with col4:
            st.metric(
                "Primary Intent",
                st.session_state.analysis_results.get('primary_intent', 'Unknown').title()
            )
        
        # Intent Analysis
        st.markdown('<div class="metrics-box">', unsafe_allow_html=True)
        st.subheader("🎯 Search Intent Analysis")
        st.write(f"**Content Type:** {st.session_state.analysis_results.get('content_type', 'Guide')}")
        st.write(f"**User Goal:** {st.session_state.analysis_results.get('user_goal', 'Information seeking')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Optimized Outline",
                data=st.session_state.optimized_outline,
                file_name=f"optimized_outline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col2:
            # Create summary report
            summary = f"""# Optimization Report
            
## Metrics
- Original Bullets: {st.session_state.optimization_metadata['original_bullets']}
- New Bullets: {st.session_state.optimization_metadata['new_bullets']}
- Expansion Rate: {st.session_state.optimization_metadata['expansion_rate']:.1f}%
- Primary Intent: {st.session_state.optimization_metadata['primary_intent']}

## Analysis
{json.dumps(st.session_state.analysis_results, indent=2)}
"""
            st.download_button(
                label="📊 Download Report",
                data=summary,
                file_name=f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        # Show optimized outline
        st.markdown("### 📝 Optimized Outline Preview")
        st.markdown(st.session_state.optimized_outline)
        
        # Success message
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success("""
        ✅ **Success!** Your outline has been optimized while preserving all original content.
        The enhanced version is ready for immediate use.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
