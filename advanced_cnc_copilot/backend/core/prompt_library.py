"""
Prompt Library 📚
Stores curated prompt presets for Manufacturing/CNC tasks.
"""
from typing import List, Dict

class PromptLibrary:
    @staticmethod
    def get_presets() -> List[Dict[str, str]]:
        return [
            {
                "id": "gcode_explain",
                "title": "Explain G-Code",
                "description": "Analyze a block of G-Code and explain its operations.",
                "prompt": "Explain the following G-Code in detail, focusing on safety and operations:\n\n```gcode\n% \nO1000\nN10 G90 G21 G17\nN20 G54 X0 Y0 S1200 M03\n... \n```"
            },
            {
                "id": "optimize_feed",
                "title": "Optimize Feed Rate",
                "description": "Calculate optimal feed rates for specific materials.",
                "prompt": "I am milling Aluminum 6061 with a 10mm 4-flute carbide end mill. My spindle speed is 8000 RPM. Calculate the optimal Feed Rate (IPM or mm/min) and explain the chip load calculation."
            },
            {
                "id": "troubleshoot_vibration",
                "title": "Troubleshoot Chatter",
                "description": "Diagnose causes of vibration or chatter during machining.",
                "prompt": "I am experiencing high-frequency chatter during a finishing pass on a steel part. \nTool stickout: 50mm\nDepth of Cut: 2mm\nwidth of cut: 0.5mm\n\nSuggest 3 potential causes and fixes."
            },
            {
                "id": "fanuc_macro",
                "title": "Generate Fanuc Macro",
                "description": "Create a parametric macro B program.",
                "prompt": "Write a Fanuc Macro B program to mill a circular pocket. \nParameters needed:\n#100 = X Center\n#101 = Y Center\n#102 = Diameter\n#103 = Depth\n#104 = Stepover"
            },
            {
                "id": "safety_check",
                "title": "Safety Compliance Check",
                "description": "Review operational parameters for safety violations.",
                "prompt": "Review this operation for safety:\nMaterial: Titanium\nCoolant: OFF\nSpeed: 15000 RPM\nFeed: 5000 mm/min\nTool: HSS Drill\n\nIdentify any risks."
            }
        ]
