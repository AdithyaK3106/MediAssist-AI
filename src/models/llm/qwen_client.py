import json
import os
import time
import socket
import urllib.request
import urllib.error
from urllib.parse import urlparse

class QwenLLMClient:
    """
    Modular client for Qwen 2.5 7B.
    Connects to Ollama on host.docker.internal:11434 when available,
    with an intelligent bilingual fallback generator for development machines.
    """
    _cached_available = None
    _last_check_time = 0

    def __init__(self, base_url=None, model="qwen2.5:7b"):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.model = model

    def is_available(self) -> bool:
        """Check if Ollama service is reachable with fast caching."""
        now = time.time()
        if QwenLLMClient._cached_available is not None and (now - QwenLLMClient._last_check_time < 30):
            return QwenLLMClient._cached_available

        try:
            parsed = urlparse(self.base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 11434
            with socket.create_connection((host, port), timeout=0.15):
                QwenLLMClient._cached_available = True
                QwenLLMClient._last_check_time = now
                return True
        except Exception:
            QwenLLMClient._cached_available = False
            QwenLLMClient._last_check_time = now
            return False

    def generate_json(self, prompt: str, system_prompt: str = None) -> dict:
        """
        Calls Ollama Qwen 2.5 7B using native JSON mode (format='json').
        Forces the LLM to output valid JSON for clinical extraction.
        """
        payload_dict = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
        if system_prompt:
            payload_dict["system"] = system_prompt

        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_text = data.get("response", "{}").strip()
            return json.loads(raw_text)

    def generate_triage_response(self, decision_bundle: dict, timeline: dict, language: str = "en") -> str:
        """
        Generate empathetic, professional triage response in English or Hindi.
        """
        if self.is_available():
            try:
                return self._generate_via_ollama(decision_bundle, timeline, language)
            except Exception as e:
                print(f"[QwenClient] Ollama call failed ({e}). Using native bilingual synthesis.")

        return self._generate_native(decision_bundle, timeline, language)

    def _generate_via_ollama(self, decision_bundle: dict, timeline: dict, language: str) -> str:
        prompt = f"""You are MediAssist AI, an empathetic healthcare triage assistant.
Generate a concise, reassuring message for the patient in {'Hindi' if language == 'hi' else 'English'}.
Strictly follow these clinical triage facts:
- Primary condition suspected: {decision_bundle.get('primary_condition')}
- Top conditions: {json.dumps(decision_bundle.get('top_conditions'))}
- Urgency: {decision_bundle.get('urgency_level')}
- Recommended timeframe: {decision_bundle.get('action_timeframe')}
- Home care tips: {json.dumps(decision_bundle.get('home_care'))}
- Red flags to watch: {json.dumps(decision_bundle.get('red_flags'))}

Guidelines:
1. Do NOT claim this is a definitive diagnosis.
2. Communicate the top conditions clearly using their clinical relative likelihood percentages and match bands.
3. Emphasize that professional confirmation is required.
4. Be respectful, clear, reassuring, and easy to read in WhatsApp format.
"""
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()

    def _generate_native(self, decision_bundle: dict, timeline: dict, language: str) -> str:
        top_conditions = decision_bundle.get("top_conditions", [])
        urgency = decision_bundle.get("urgency_level", "Moderate")
        action_timeframe = decision_bundle.get("action_timeframe", "")
        home_care = decision_bundle.get("home_care", [])
        red_flags = decision_bundle.get("red_flags", [])
        hospitals = decision_bundle.get("hospitals", [])

        if language == "hi":
            # Hindi Response Format
            urgency_hi = {"Low": "सामान्य (Low)", "Moderate": "मध्यम (Moderate)", "High": "गंभीर (High)"}.get(urgency, urgency)
            out = f"नमस्ते। आपके बताए गए लक्षणों और समय-सीमा के आधार पर यह प्रारंभिक स्वास्थ्य विश्लेषण है:\n\n"
            out += f"📊 **प्राथमिक संभावित स्थितियां:**\n"
            for i, c in enumerate(top_conditions[:3]):
                out += f"{i+1}. **{c['condition']}** — {c.get('match_band_hi', 'संभावना')}: {c['confidence']}% सापेक्ष संभावना\n"
            
            out += f"\n⏱️ **आवश्यकता स्तर (Urgency):** {urgency_hi}\n"
            out += f"👉 **सलाह:** {action_timeframe}\n"

            if home_care:
                out += f"\n🏡 **सहायक घरेलू देखभाल (Home Care):**\n"
                for tip in home_care:
                    out += f"• {tip}\n"
            else:
                out += f"\n⚠️ **कृपया ध्यान दें:** इस स्थिति में घरेलू उपचार सुरक्षित नहीं है। सीधे डॉक्टर से परामर्श करें।\n"

            if red_flags:
                out += f"\n🚩 **डॉक्टर को कब तुरंत दिखाएं (Red Flags):**\n"
                for rf in red_flags:
                    out += f"• {rf}\n"

            if hospitals:
                out += f"\n🏥 **निकटतम अनुशंसित अस्पताल (बैंगलोर):**\n"
                for h in hospitals[:2]:
                    out += f"• **{h['name']}** ({h['city']}) — दूरी: {h['distance']} किमी\n"

            out += "\n\n⚠️ **चिकित्सा अस्वीकरण (Disclaimer):** यह केवल एक प्रारंभिक कंप्यूटर-जनित सुझाव है, कोई अंतिम चिकित्सा निदान नहीं। कृपया किसी योग्य डॉक्टर से संपर्क करें।"
        else:
            # English Response Format
            out = f"Hello. Based on your symptoms and timeline, here is your preliminary triage assessment:\n\n"
            out += f"📊 **Top Possible Conditions:**\n"
            for i, c in enumerate(top_conditions[:3]):
                out += f"{i+1}. **{c['condition']}** — {c.get('match_band', 'Match')}: {c['confidence']}% relative likelihood\n"
            
            out += f"\n⏱️ **Urgency Level:** {urgency}\n"
            out += f"👉 **Recommended Action:** {action_timeframe}\n"

            if home_care:
                out += f"\n🏡 **Supportive Home Care:**\n"
                for tip in home_care:
                    out += f"• {tip}\n"
            else:
                out += f"\n⚠️ **Caution:** Home remedies are not advised for this severity level. Direct medical consultation is needed.\n"

            if red_flags:
                out += f"\n🚩 **When to Seek Immediate Care:**\n"
                for rf in red_flags:
                    out += f"• {rf}\n"

            if hospitals:
                out += f"\n🏥 **Nearest Hospitals (Bangalore):**\n"
                for h in hospitals[:2]:
                    out += f"• **{h['name']}** ({h['city']}) — {h['distance']} km away\n"

            out += "\n\n⚠️ **Medical Disclaimer:** MediAssist AI provides preliminary guidance only, not a clinical diagnosis. Always consult a qualified medical professional for confirmation."

        return out
