"""Build the public three-page report; contains no private clinical records."""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
DRIVE = 'https://drive.google.com/drive/folders/1ekAAR7IXt-S8xGq9QeSjdZR-iDEuiEUx?usp=sharing'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyClean', fontName='Helvetica', fontSize=10.5, leading=15, spaceAfter=10))
styles.add(ParagraphStyle(name='CodeClean', fontName='Courier', fontSize=8, leading=12, spaceAfter=9))
styles['Title'].textColor = colors.HexColor('#16394b')
styles['Heading2'].textColor = colors.HexColor('#166c72')


def p(text, style='BodyClean'):
    return Paragraph(text, styles[style])


def section(title, body):
    return [p(title, 'Heading2'), p(body)]


def footer(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#526570'))
    canvas.drawString(42, 27, 'TermMap | HiLabs AI Quest 2026 | Public-release report')
    canvas.drawRightString(A4[0]-42, 27, f'Page {doc.page} of 3')


def main():
    output = ROOT / 'output/pdf/termmap_submission_report.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=38, bottomMargin=45, title='TermMap - HiLabs AI Quest 2026',
                            author='TermMap submission', subject='Public-release methodology and limitations')
    story = [p('TermMap', 'Title'), p('Local ICD-10-CM and SNOMED CT mapping', 'Heading2')]
    story += section('Problem and submitted scope',
        'Given an extracted diagnosis or condition, an identifier and optional nearby text, return one code from each authorized terminology workbook. The submitted path runs locally without a hosted API. Entity extraction and non-diagnosis categories are outside this scope.')
    story += section('Release contents',
        'Source code, dependency specifications, synthetic tests and inputs, this report and a captioned demo are included. No pretrained fitted artifacts, proprietary training records, terminology dictionaries, evaluation predictions or credentials are distributed. Authorized reviewers must provide private inputs and train locally.')
    story += section('Pipeline and rationale',
        'Normalize Unicode, punctuation and stable abbreviations. Retrieve candidates independently per system with SQLite FTS5 and character/word/context TF-IDF over fitting examples. Two LightGBM rankers combine lexical similarity, context, qualifiers and frequency. Check selected codes against the supplied dictionaries and write the required CSV. No crosswalk supplies the target codes.')
    story += [p('Historical grouped holdout results', 'Heading2')]
    rows = [['Path / split', 'ICD exact', 'SNOMED exact'],
            ['Local / seed 42', '226/471 (48.0%)', '260/537 (48.4%)'],
            ['Local / seed 43', '238/451 (52.8%)', '269/546 (49.3%)'],
            ['Optional DeepSeek 1017 / seed 42', '307/471 (65.2%)', '328/537 (61.1%)']]
    table = Table([[p(x) for x in row] for row in rows], colWidths=[245,132,132])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5f0f2')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('BOX',(0,0),(-1,-1),0.5,colors.lightgrey),
                              ('INNERGRID',(0,0),(-1,-1),0.25,colors.lightgrey),
                              ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story += [table, Spacer(1,12), p('These figures come from the original submission documentation and recording. They were not rerun or independently verified during the release audit; private row-level evidence is excluded. No documented path reaches the 95% target. Hosted results are experimental and are not the submitted default predictor.'), PageBreak()]
    story += [p('Evaluation and limitations', 'Title')]
    story += section('Protocol and interpretation',
        'Validation groups normalized entities across coding systems into an 80/20 fitting/holdout split. Development choices used seed 42, so that result is not an untouched final test. Entity grouping does not establish patient-level or institution-level independence. The figures are not hidden challenge evaluation results.')
    story += section('Failure categories',
        'Retrieval may miss rare or unseen codes. Ranking can confuse related concepts and specificity levels even when the correct candidate is present. Ambiguous abbreviations, spelling noise, negation, history and laterality remain weaknesses. Candidate scores are ranking signals, not calibrated probabilities. Dictionary membership does not establish clinical correctness or billing validity.')
    story += section('Synthetic demonstration',
        'The 151-second silent, captioned video uses the synthetic examples/hard_cases.csv. Its abbreviation case shows a historical local model selecting an altered-mental-status code despite neurological context. This is a synthetic illustration, not a private holdout quotation or evidence of accuracy. Fitted artifacts used in the recording are not included; new fits may differ.')
    story += section('Optional research paths',
        'Source for SapBERT dense retrieval, BOND reranking, learned adaptation and OpenRouter adjudication is preserved. These paths require additional dependencies, model downloads or external services. They are excluded from the default submission command and were not exercised during release verification. Neural or hosted overrides can also regress correct local choices.')
    story += section('Next steps and intended use',
        'Future work would improve candidate recall, validate ranking on fresh groups, and review ambiguous labels with qualified coders. These are proposals, not new experiments or performance claims. Clinical deployment, calibration, fairness and prospective validation are not established. Outputs require qualified review before clinical or billing use.')
    story += [PageBreak(), p('Reproducibility and privacy', 'Title')]
    story += section('Installation and inputs',
        'Use Python 3.12 or 3.13 with SQLite FTS5. Install requirements.txt in a virtual environment. Place authorized hackathon_train.csv, icd10cm_codes.xlsx and SNOMEDCT_US_codes_and_descriptions.xlsx in private_data/. The README describes the full training and workbook schemas.')
    for cmd in ['python3 -m venv .venv', '.venv/bin/python -m pip install -r requirements.txt',
                '.venv/bin/python -m termmap.cli --data-dir private_data train',
                '.venv/bin/python -m termmap.cli --data-dir private_data predict',
                '  --input private_data/evaluation.csv --output artifacts/predictions.csv']:
        story.append(p(escape(cmd), 'CodeClean'))
    story += [p('The final two displayed lines are one prediction command. Missing or invalidated indexes/rankers are built automatically, so a first prediction can train. Retain source files for cache validation. No fitted artifacts are shipped.')]
    story += section('CSV contract',
        'Input requires clinical_entity (alias entity), with optional chunk_id and chunk_text (aliases context or text). Missing IDs use one-based row numbers; empty entities are rejected. Output preserves order with exactly chunk_id, clinical_entity, icd10cm_code, snomed_code.')
    story += section('Privacy and verification',
        'Default inference is local. Fitted objects retain training examples; keep all artifacts, source data, diagnostics, predictions and credentials private. Hosted adjudication is opt-in and sends text to a provider; requested retention settings are not independent guarantees. Release checks use synthetic fixtures and mocked requests, with no training, model downloads or hosted calls. See PRIVACY_REVIEW.md for actual checks and skips.')
    story += [p('Demo and supporting assets', 'Heading2'),
              p('The recording is output/video/termmap_demo.mp4; DEMO.md provides the runbook. README.md and MODEL_CARD.md document setup, historical results and limitations.'),
              p(f'<link href="{escape(DRIVE)}" color="#166c72">Supporting Google Drive folder</link>'),
              p('Replace the earlier Drive PDF with this sanitized three-page report. Drive contents and sharing permissions are separate from this source release.')]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output.relative_to(ROOT))


if __name__ == '__main__':
    main()
