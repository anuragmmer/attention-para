# Audio Attention Prediction Analysis Tool  

## Purpose  
This tool measures how likely an audio file is to capture and hold someone's attention based purely on its sound characteristics. It uses neuroscience research to analyze features that naturally grab human attention. The tool doesn't consider popularity or trends - it only looks at the technical aspects of the sound itself.  

Key benefits:  
- Applies real neuroscience to everyday audio  
- Measures specific sound features scientifically linked to attention  
- Creates easy-to-understand reports  
- Works the same for all genres and styles  

## Who Should Use This  
- Video creators: Improve openings and key moments  
- Podcasters/Musicians: Compare different versions of their work  
- Researchers: Study what makes sounds engaging  
- Teachers: Make educational content more compelling  

## What to Keep in Mind  
- Scores aren't affected by views or likes  
- Scores compare parts within the same file (a 70 in one file isn't the same as 70 in another)  
- Only analyzes sound - doesn't consider visuals or story  
- Real human attention still varies by individual  

## The Science Behind It  

The tool examines six key sound features proven by brain research to affect attention:  

1. **Sound Roughness** - When volume fluctuates quickly (40-80 times per second), it creates an attention-grabbing effect our brains find noticeable.  

2. **Sharp Sounds** - Sudden sound starts (like drum hits) make our brains react strongly.  

3. **Volume Changes** - Big, sudden volume jumps grab more attention than gradual changes.  

4. **Important Frequencies** - High tones (2,000-5,000 Hz) feel urgent, while very low tones (30-40 Hz) can create tension.  

5. **Mini-Silences** - Tiny gaps in sound (as short as 6 milliseconds) help reset attention.  

6. **Stereo Movement** - Sounds that move between left and right speakers engage listeners more.  

The tool combines these factors into easy scores:  
- Main factors (roughness, sharp sounds, volume changes) = 70% of score  
- Supporting factors (frequencies, silences, movement) = 30% of score  
- Final score from 0-100  

## How to Set Up  

### What You Need  
First, install these Python packages:  
numpy, scipy, librosa, matplotlib, plotly, pydub, weasyprint, kaleido  

You'll also need:  
- FFmpeg for handling audio files  
- WeasyPrint requirements for PDF reports  

### Installation Steps  
1. Create a virtual environment:  
   `python -m venv audio_attention_env`  

2. Activate it:  
   - Mac/Linux: `source audio_attention_env/bin/activate`  
   - Windows: `audio_attention_env\Scripts\activate`  

3. Install requirements:  
   `pip install -r requirements.txt`  

## How to Use  

1. Run the tool: `python att.py`  
2. Select your audio file (supports mp3, wav, flac, m4a, ogg)  
3. Wait 30 seconds to 2 minutes for analysis  
4. Get your results in a new folder containing:  
   - Interactive HTML report  
   - PDF summary  
   - Raw data JSON file  

## Understanding Your Results  

The reports show:  
- Sound waves over time (more variation usually means more engaging)  
- When roughness peaks occur  
- Where sharp sounds and volume jumps happen  
- Which frequency ranges are prominent  
- Locations of attention-resetting silences  
- Stereo movement patterns  

Score ranges:  
- 70-100: Highly engaging audio  
- 50-69: Moderately engaging  
- 0-49: Less engaging  

## Real-World Uses  

- YouTube creators: Make better openings to keep viewers watching  
- Podcasters: Find and fix boring sections  
- Musicians: Test which versions of songs hold attention best  
- Teachers: Record more engaging lectures  

## Research References  

The tool is based on studies from:  
- Johns Hopkins University  
- Nature Communications  
- Proceedings of the National Academy of Sciences  
- Frontiers in Neuroscience  
- And 20+ other peer-reviewed sources  

(Complete reference list available in the references documentation)  

This plain text version keeps all the key information while removing formatting for maximum readability. You can copy this directly or use it as a base for your README file.
