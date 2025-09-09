
import time
import json
from pathlib import Path
import logging

from m3u8_to_mp3 import download_and_convert_m3u8
from audio_cleaning import clean_audio
from clean_audio_to_text import transcribe_audio
from text_nlp import TextProcessor
from accuracy import ProtocolComplianceChecker

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleTutorAnalyzer:
    def __init__(self):
        self.output_dir = Path("output")
        self.audio_dir = self.output_dir / "audio"
        self.transcription_dir = self.output_dir / "transcriptions"
        self.reports_dir = self.output_dir / "reports"

        self._make_dirs()

        self.text_processor = TextProcessor()
        self.compliance_checker = ProtocolComplianceChecker()

    def _make_dirs(self):
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcription_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, m3u8_url: str):
        session_id = f"session_{int(time.time())}"

        raw_audio_path = self.audio_dir / f"{session_id}.mp3"
        cleaned_audio_path = self.audio_dir / f"{session_id}_cleaned.wav"
        transcription_path = self.transcription_dir / f"{session_id}.txt"
        report_path = self.reports_dir / f"{session_id}_report.txt"

        try:
            logger.info("Step 1: Downloading and converting audio...")
            download_and_convert_m3u8(m3u8_url, str(raw_audio_path))

            logger.info("Step 2: Cleaning audio...")
            clean_audio(str(raw_audio_path), str(cleaned_audio_path))

            logger.info("Step 3: Transcribing audio...")
            transcription = transcribe_audio(
                str(cleaned_audio_path),
                model_size="medium",
                output_txt=str(transcription_path),
                verbose=False
            )

            logger.info("Step 4: Preprocessing text...")
            processed_text = self.text_processor.preprocess(transcription["text"])

            logger.info("Step 5: Analyzing protocol compliance...")
            analysis = self.compliance_checker.analyze_text(processed_text)

            report = self.compliance_checker.generate_report(analysis)

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            logger.info(f"✅ Report saved to: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"❌ Error during analysis: {e}", exc_info=True)
            return None




if __name__ == "__main__":
    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/d8862f7f16d0ae0aa5b14d1542406054/f30e1bfdd886b6b7d0c481d1b8916c8d/stream/playlist.m3u8"
    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/c89a4513f49f5bcff3d97d5110f8aab9/2f841ca0ab94462cf078dc614bda98dd/stream/playlist.m3u8" #1

    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/59cc900838c042c225d56d5070c915c5/4121e0a666022a2824682fd34d821578/stream/playlist.m3u8" #2 
    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/ca1711c5453c25d9970db8664fab4c8d/38ff3a44add8711c2bc2bad6f652c0a9/stream/playlist.m3u8" #3
    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/254c8c5e20c68d572ac1a32923770557/61d8ff65434d84e9e14bf886afdb8289/stream/playlist.m3u8"
    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/fcbd0ec10f36ea37daa6e5e2040ea642/ac62c655ec5e6c91620361a10d2fed38/stream/playlist.m3u8"
    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/ef308663b7e1bb63cd31d6ee18e8cc03/dd235fc6c41a45fd1362a85455fa74f1/stream/playlist.m3u8"
    # m3u8_url ="https://filo-classroom-video.s3.us-east-005.backblazeb2.com/d17a6f93d8fabe7e8706e850ab20726b/f9fa7e7dd2b06924cfd8045d8e565342/stream/playlist.m3u8"
    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/17cb36957191c104a3b62de78a4a6db2/66f4cbdfc9896dc16d84fb3fa44753a3/stream/playlist.m3u8"
    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/ec3ba8b0756986f0b6a239117c2fc165/0bf50f858ad0d98cef85f782d72aec90/stream/playlist.m3u8" #4
    # m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/d7944fc4df9d0d75018bcae353fc352c/ff5a78ecf25d7d809754543db4cbc3ea/stream/playlist.m3u8"
    m3u8_url = "https://filo-classroom-video.s3.us-east-005.backblazeb2.com/56233b3a6401700d42bbe98f4445c664/6d8c601ca620d9f167772521e7ebc03e/stream/playlist.m3u8"
    analyzer = SimpleTutorAnalyzer()
    report = analyzer.analyze(m3u8_url)

    if report:
        print(f"\n Done! Report saved at: {report}")
    else:
        print("\n Something went wrong.")
