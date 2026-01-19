import argparse
import subprocess
import os
from datetime import datetime

def run_command(command):
    """Runs a command and prints its output."""
    print(f"Running command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(process.stderr)
        raise RuntimeError(f"Command failed: {' '.join(command)}")
    print(process.stdout)

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Create a company slideshow with metrics and audio.")
    parser.add_argument("--audio_file", required=True, help="Filepath of the audio file.")
    parser.add_argument("--audio_file_chinese", help="Filepath of the correspondent Chinese audio file.")
    parser.add_argument("--sticker", required=True, help="Sticker of the company stock.")
    parser.add_argument("--title", required=True, help="Title of the slides.")
    parser.add_argument("--title_chinese", help="Chinese title of the slides.")
    parser.add_argument("--output-dir", default="output", help="Output directory for all generated files.")
    args = parser.parse_args()

    sticker = args.sticker.upper()
    title = args.title
    title_chinese = args.title_chinese
    audio_file = args.audio_file
    audio_file_chinese = args.audio_file_chinese
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    current_year = datetime.now().year
    start_year = current_year - 10

    # 1. Generate the CSV table for the last 10 years and the current year.
    sec_metrics_output_dir = os.path.join(output_dir, "sec_metrics_output", sticker)
    sec_metrics_command = [
        "python",
        "sec_metrics_builder/sec_metrics_builder.py",
        "--ticker",
        sticker,
        "--years",
        str(start_year),
        str(current_year),
        "--output-dir",
        sec_metrics_output_dir,
        "--core-metrics-only"
    ]
    run_command(sec_metrics_command)

    # 2. Generate the metrics charts.
    charts_output_dir = os.path.join(output_dir, sticker)
    generate_charts_command = [
        "python",
        "generate_metric_charts.py",
        "--input",
        os.path.join(sec_metrics_output_dir, f"{sticker}.csv"),
        "--output",
        charts_output_dir,
        "--zip",
        os.path.join(output_dir, "metric_charts.zip")
    ]
    run_command(generate_charts_command)

    # 3. Create the English video.
    slideshow_command_en = [
        "./make_slideshow_slides_oss.sh",
        "--file",
        audio_file,
        "--output-dir",
        output_dir,
        "30", # CHUNK (default)
        title, # TITLE
        charts_output_dir, # IMG_DIR
        "English", # LANG
        "1.02" # SPEED
    ]
    run_command(slideshow_command_en)

    # 4. Create the Chinese video if an audio file is provided.
    if audio_file_chinese:
        # Generate Chinese charts
        charts_output_dir_zh = os.path.join(output_dir, f"{sticker}_zh")
        generate_charts_command_zh = [
            "python",
            "generate_metric_charts.py",
            "--input",
            os.path.join(sec_metrics_output_dir, f"{sticker}.csv"),
            "--output",
            charts_output_dir_zh,
            "--language",
            "chinese",
            "--font",
            "SourceHanSansSC-Bold.otf",
            "--zip",
            os.path.join(output_dir, "metric_charts_zh.zip")
        ]
        run_command(generate_charts_command_zh)

        translated_title = title_chinese if title_chinese else f"中文 - {title}"

        slideshow_command_zh = [
            "./make_slideshow_slides_oss.sh",
            "--file",
            audio_file_chinese,
            "--output-dir",
            output_dir,
            "30", # CHUNK (default)
            translated_title, # TITLE
            charts_output_dir_zh, # IMG_DIR
            "Chinese", # LANG
            "1.08" # SPEED
        ]
        run_command(slideshow_command_zh)

if __name__ == "__main__":
    main()
