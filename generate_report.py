"""
Generate Training Report PDF for Liver Fibrosis Staging Model.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
from pathlib import Path

# Output path
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUTPUT_DIR / "training_report.pdf"

def create_training_report():
    """Generate the training report PDF."""
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
        spaceAfter=30,
        textColor=HexColor('#1a365d'),
        alignment=1  # Center
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
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        textColor=HexColor('#4a5568')
    )
    
    # Title
    story.append(Paragraph("🏥 Liver Fibrosis Staging", title_style))
    story.append(Paragraph("Training Completion Report", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
    story.append(Spacer(1, 30))
    
    # Model Summary
    story.append(Paragraph("📊 Model Summary", heading_style))
    
    model_data = [
        ["Parameter", "Value"],
        ["Model Architecture", "Vision Transformer (ViT-B/16)"],
        ["Pre-trained Weights", "ImageNet-1K"],
        ["Input Size", "224 × 224 pixels"],
        ["Number of Classes", "5 (F0, F1, F2, F3, F4)"],
        ["Total Parameters", "~86 Million"],
    ]
    
    model_table = Table(model_data, colWidths=[2.5*inch, 3.5*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3182ce')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
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
    story.append(model_table)
    story.append(Spacer(1, 20))
    
    # Training Configuration
    story.append(Paragraph("⚙️ Training Configuration", heading_style))
    
    config_data = [
        ["Setting", "Value"],
        ["Device", "NVIDIA GeForce RTX 3050 (CUDA)"],
        ["Total Epochs", "20"],
        ["Batch Size", "8"],
        ["Learning Rate", "0.0001"],
        ["Optimizer", "AdamW (weight_decay=0.01)"],
        ["Scheduler", "CosineAnnealingLR"],
        ["Loss Function", "CrossEntropyLoss (label_smoothing=0.1)"],
        ["Early Stopping", "Patience = 5"],
    ]
    
    config_table = Table(config_data, colWidths=[2.5*inch, 3.5*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0fff4')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2d3748')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#9ae6b4')),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 20))
    
    # Dataset Information
    story.append(Paragraph("📁 Dataset Information", heading_style))
    
    dataset_data = [
        ["Metric", "Value"],
        ["Total Samples", "6,323 images"],
        ["Training Samples", "5,058 (80%)"],
        ["Validation Samples", "1,265 (20%)"],
        ["Class F0", "2,114 images"],
        ["Class F1", "861 images"],
        ["Class F2", "793 images"],
        ["Class F3", "857 images"],
        ["Class F4", "1,698 images"],
    ]
    
    dataset_table = Table(dataset_data, colWidths=[2.5*inch, 3.5*inch])
    dataset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#805ad5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#faf5ff')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2d3748')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#d6bcfa')),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(dataset_table)
    story.append(Spacer(1, 20))
    
    # Training Results - THE MAIN SECTION
    story.append(Paragraph("🏆 Training Results", heading_style))
    
    results_data = [
        ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Status"],
        ["1", "1.1734", "53.58%", "1.1760", "51.94%", "✓"],
        ["5", "0.8948", "70.84%", "0.8945", "70.67%", "✓"],
        ["10", "0.7055", "80.24%", "0.7398", "79.84%", "✓"],
        ["12", "0.6428", "86.24%", "0.6263", "87.11%", "✓"],
        ["15", "0.5335", "93.12%", "0.5330", "92.96%", "✓"],
        ["18", "0.4615", "96.68%", "0.5032", "95.10%", "★ BEST"],
        ["20", "0.4450", "97.37%", "0.4876", "95.02%", "✓"],
    ]
    
    results_table = Table(results_data, colWidths=[0.7*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#dd6b20')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#fffaf0')),
        ('BACKGROUND', (0, 6), (-1, 6), HexColor('#c6f6d5')),  # Highlight best epoch
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2d3748')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#fbd38d')),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 30))
    
    # Final Performance Summary
    story.append(Paragraph("🎯 Final Performance Summary", heading_style))
    
    final_style = ParagraphStyle(
        'FinalResult',
        parent=styles['Normal'],
        fontSize=18,
        spaceAfter=10,
        textColor=HexColor('#276749'),
        alignment=1,
        fontName='Helvetica-Bold'
    )
    
    story.append(Paragraph("Best Validation Accuracy: 95.10%", final_style))
    story.append(Paragraph("Best Training Accuracy: 97.37%", final_style))
    story.append(Paragraph("Total Training Time: 95.8 minutes", final_style))
    story.append(Spacer(1, 20))
    
    # Model File Info
    story.append(Paragraph("📦 Model Checkpoint", heading_style))
    story.append(Paragraph("Location: D:\\ALS\\outputs\\vit_light\\best_vit_model.pth", body_style))
    story.append(Paragraph("Repository: github.com/RangeshPandianPT/Liver-fibrosis-staging", body_style))
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
    print(f"✅ PDF Report generated: {PDF_PATH}")
    return PDF_PATH

if __name__ == "__main__":
    create_training_report()
