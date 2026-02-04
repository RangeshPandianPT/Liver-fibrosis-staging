"""
Generate Evaluation Report PDF for ResNet50 CNN Model.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from datetime import datetime
from pathlib import Path

# Output path
OUTPUT_DIR = Path(__file__).parent / "outputs" / "metrics" / "cnn_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUTPUT_DIR / "resnet50_evaluation_report.pdf"

def create_evaluation_report():
    """Generate the ResNet50 evaluation report PDF."""
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        textColor=HexColor('#1a365d'),
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=20,
        textColor=HexColor('#4a5568'),
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=HexColor('#2d3748')
    )
    
    # Title
    story.append(Paragraph("🧠 ResNet50 Evaluation Report", title_style))
    story.append(Paragraph("Liver Fibrosis Staging - CNN Model Performance", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
    story.append(Spacer(1, 30))
    
    # Overall Metrics
    story.append(Paragraph("📊 Overall Performance Metrics", heading_style))
    
    overall_data = [
        ["Metric", "Value"],
        ["Accuracy", "91.30%"],
        ["F1-Score (Macro)", "0.8766"],
        ["F1-Score (Weighted)", "0.9118"],
        ["Precision (Macro)", "0.8892"],
        ["Recall (Macro)", "0.8708"],
        ["Cohen's Kappa", "0.9751"],
        ["ROC-AUC (Macro)", "0.9889"],
    ]
    
    overall_table = Table(overall_data, colWidths=[2.5*inch, 2.5*inch])
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3182ce')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#edf2f7')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2d3748')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#cbd5e0')),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(overall_table)
    story.append(Spacer(1, 25))
    
    # Per-Class F1-Scores
    story.append(Paragraph("📈 Per-Class F1-Scores", heading_style))
    
    f1_data = [
        ["Class", "F1-Score", "Performance"],
        ["F0 (No Fibrosis)", "1.0000", "⭐ Excellent"],
        ["F1 (Mild)", "0.8545", "✓ Good"],
        ["F2 (Moderate)", "0.7751", "○ Moderate"],
        ["F3 (Severe)", "0.8039", "✓ Good"],
        ["F4 (Cirrhosis)", "0.9493", "⭐ Excellent"],
    ]
    
    f1_table = Table(f1_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    f1_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, 1), HexColor('#c6f6d5')),  # F0 - green
        ('BACKGROUND', (0, 2), (-1, 2), HexColor('#f0fff4')),
        ('BACKGROUND', (0, 3), (-1, 3), HexColor('#fefcbf')),  # F2 - yellow
        ('BACKGROUND', (0, 4), (-1, 4), HexColor('#f0fff4')),
        ('BACKGROUND', (0, 5), (-1, 5), HexColor('#c6f6d5')),  # F4 - green
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2d3748')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#9ae6b4')),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(f1_table)
    story.append(Spacer(1, 25))
    
    # Per-Class Accuracy
    story.append(Paragraph("🎯 Per-Class Accuracy", heading_style))
    
    acc_data = [
        ["Class", "Accuracy", "Status"],
        ["F0 (No Fibrosis)", "100.00%", "⭐ Perfect"],
        ["F1 (Mild)", "81.98%", "✓ Good"],
        ["F2 (Moderate)", "82.39%", "✓ Good"],
        ["F3 (Severe)", "71.93%", "△ Needs Improvement"],
        ["F4 (Cirrhosis)", "99.12%", "⭐ Excellent"],
    ]
    
    acc_table = Table(acc_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    acc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#805ad5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, 1), HexColor('#e9d8fd')),
        ('BACKGROUND', (0, 2), (-1, 2), HexColor('#faf5ff')),
        ('BACKGROUND', (0, 3), (-1, 3), HexColor('#faf5ff')),
        ('BACKGROUND', (0, 4), (-1, 4), HexColor('#fed7d7')),  # F3 - highlight concern
        ('BACKGROUND', (0, 5), (-1, 5), HexColor('#e9d8fd')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2d3748')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#d6bcfa')),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(acc_table)
    story.append(Spacer(1, 30))
    
    # Key Findings
    story.append(Paragraph("💡 Key Findings", heading_style))
    
    findings_style = ParagraphStyle(
        'Findings',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        leftIndent=20,
        textColor=HexColor('#4a5568')
    )
    
    findings = [
        "• <b>Overall Accuracy of 91.30%</b> demonstrates strong model performance",
        "• <b>Cohen's Kappa of 0.9751</b> indicates near-perfect agreement beyond chance",
        "• <b>ROC-AUC of 0.9889</b> shows excellent discrimination capability",
        "• <b>F0 and F4 classes</b> achieve near-perfect classification (100% and 99.12%)",
        "• <b>F3 class at 71.93%</b> is the most challenging to classify correctly",
        "• Model performs best on extreme stages (F0, F4) vs intermediate stages (F1-F3)",
    ]
    
    for finding in findings:
        story.append(Paragraph(finding, findings_style))
    
    story.append(Spacer(1, 30))
    
    # Model Info
    story.append(Paragraph("📦 Model Information", heading_style))
    
    model_data = [
        ["Parameter", "Value"],
        ["Architecture", "ResNet50"],
        ["Pre-trained Weights", "ImageNet-1K"],
        ["Checkpoint Location", "D:\\ALS\\outputs\\checkpoints\\"],
        ["Evaluation Results", "D:\\ALS\\outputs\\metrics\\cnn_evaluation\\"],
    ]
    
    model_table = Table(model_data, colWidths=[2*inch, 4*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#dd6b20')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#fffaf0')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2d3748')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#fbd38d')),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 30))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#718096'),
        alignment=1
    )
    story.append(Paragraph("─" * 60, footer_style))
    story.append(Paragraph("Generated by Liver Fibrosis Staging Pipeline", footer_style))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    
    # Build PDF
    doc.build(story)
    print(f"✅ ResNet50 Evaluation Report generated: {PDF_PATH}")
    return PDF_PATH

if __name__ == "__main__":
    create_evaluation_report()
