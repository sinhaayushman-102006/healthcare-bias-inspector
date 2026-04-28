from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(filename, acc_dict, bias_gap):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Healthcare AI Bias Report", styles["Title"]))

    for group, acc in acc_dict.items():
        content.append(Paragraph(f"{group}: {acc:.2f}", styles["Normal"]))

    content.append(Paragraph(f"Bias Gap: {bias_gap:.2f}", styles["Normal"]))

    doc.build(content)