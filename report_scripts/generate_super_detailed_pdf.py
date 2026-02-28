import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime

# Initialize styles
styles = getSampleStyleSheet()

# Custom styles for the detailed report
title_style = ParagraphStyle(
    'MainTitle',
    parent=styles['Heading1'],
    fontSize=24,
    spaceAfter=20,
    textColor=colors.darkblue,
    alignment=1 # Center alignment
)

heading2_style = ParagraphStyle(
    'Heading2',
    parent=styles['Heading2'],
    fontSize=18,
    spaceBefore=15,
    spaceAfter=10,
    textColor=colors.HexColor('#1E88E5')
)

heading3_style = ParagraphStyle(
    'Heading3',
    parent=styles['Heading3'],
    fontSize=14,
    spaceBefore=10,
    spaceAfter=5,
    textColor=colors.darkslategrey
)

body_style = ParagraphStyle(
    'BodyText',
    parent=styles['BodyText'],
    fontSize=11,
    leading=15,
    spaceAfter=10
)

bullet_style = ParagraphStyle(
    'BulletList',
    parent=styles['BodyText'],
    fontSize=11,
    leading=15,
    leftIndent=20,
    spaceAfter=5,
    bulletIndent=10
)

def build_pdf(filename="d:/ALS/outputs/Detailed_Project_Report.pdf"):
    """Generates the super detailed project report PDF."""
    
    # Create the doc template
    doc = SimpleDocTemplate(
        filename, 
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Story holds all elements (paragraphs, tables, etc) linearly
    story = []
    
    # ----- Title Page -----
    story.append(Spacer(1, 100))
    story.append(Paragraph("ALS: Automated Liver Staging Project", title_style))
    story.append(Paragraph("A Deep Learning Approach to Non-Invasive Liver Fibrosis Assessment", ParagraphStyle('SubTitle', parent=styles['Heading2'], alignment=1, textColor=colors.gray)))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", ParagraphStyle('DateStr', parent=styles['Normal'], alignment=1)))
    story.append(PageBreak())
    
    # ----- 1. Project Overview -----
    story.append(Paragraph("1. Project Overview", heading2_style))
    story.append(Paragraph(
        "The Automated Liver Staging (ALS) project aims to revolutionize the diagnosis and staging of liver fibrosis. "
        "Traditionally, liver fibrosis staging requires an invasive liver biopsy, which is painful, expensive, and subject to high inter-observer variability among pathologists. "
        "The ALS project provides a non-invasive, highly accurate, computational alternative by classifying ultrasound or standard histological images into five distinct stages (F0 through F4) of fibrosis.", 
        body_style
    ))
    story.append(Paragraph(
        "By utilizing a multi-faceted deep learning ensemble, the ALS architecture identifies intricate patterns corresponding to fibrotic tissue deposition—ranging from F0 (no fibrosis) to F4 (cirrhosis). "
        "This project acts as a functional \"AI Pathologist\", presenting an interactive live dashboard that highlights feature enhancements and offers diagnostic confidence metrics.",
        body_style
    ))
    
    # Stage breakdown
    story.append(Paragraph("Fibrosis Stages Analyzed:", heading3_style))
    stages = [
        "F0: No active fibrosis. Normal liver architecture.",
        "F1: Portal fibrosis without septa. Early onset.",
        "F2: Portal fibrosis with rare septa. Moderate progression.",
        "F3: Numerous septa without cirrhosis. Severe fibrosis bridging portal tracts.",
        "F4: Cirrhosis. Widespread architectural distortion and nodule formation."
    ]
    for stage in stages:
        story.append(Paragraph(f"&bull; {stage}", bullet_style))
    story.append(Spacer(1, 10))

    # ----- 2. Advanced Ensemble Architecture -----
    story.append(Paragraph("2. Advanced Ensemble Architecture", heading2_style))
    story.append(Paragraph(
        "At the core of the ALS diagnostic engine is a heterogeneous 4-model ensemble. "
        "It employs a Soft-Voting strategy, merging the continuous softmax probabilities of diverse neural network topologies to form a stabilized final prediction. "
        "This architectural diversity counteracts individual model biasses, merging local translation-invariant feature extraction with global contextual attention.",
        body_style
    ))

    # Model Details
    story.append(Paragraph("Model 1: ConvNeXt V2 (Base Weight: 1.2)", heading3_style))
    story.append(Paragraph(
        "ConvNeXt V2 acts as the primary visual backbone. It modernizes standard ConvNets by mimicking Vision Transformer scaling capabilities while retaining the raw efficiency of convolutions. "
        "It excels at detecting localized fibrotic textures and cellular anomalies. Given its high standalone validation accuracy, it is granted the highest voting weight.",
        body_style
    ))

    story.append(Paragraph("Model 2: MedNeXt (Base Weight: 1.1)", heading3_style))
    story.append(Paragraph(
        "MedNeXt is a specialized medical imaging adaptation. It scales the receptive field to capture medically relevant anomalies without overwhelming computational limits. "
        "It has been fine-tuned extensively on nuanced medical datasets, ensuring the ensemble remains grounded in clinical histopathological features.",
        body_style
    ))

    story.append(Paragraph("Model 3: DeiT - Vision Transformer (Base Weight: 1.0)", heading3_style))
    story.append(Paragraph(
        "Data-efficient Image Transformers (DeiT) discard convolutions for self-attention mechanisms. "
        "DeiT divides the image into distinct patches and calculates the relationships between every patch simultaneously, allowing the network to understand the *global* geometry of the liver tissue rather than just local edges. "
        "This global analysis acts as the perfect counterbalance to the ConvNets.",
        body_style
    ))

    story.append(Paragraph("Model 4: ResNet-50 Baseline (Base Weight: 1.0)", heading3_style))
    story.append(Paragraph(
        "ResNet-50 provides the foundational, deeply propagated feature extraction. Using residual connections, it bypasses the vanishing gradient problem, allowing complex abstractions to form in deeper layers. "
        "As a universally recognized standard, its inclusion anchors the ensemble, ensuring extreme reliability and preventing catastrophic failures on out-of-distribution inputs.",
        body_style
    ))
    story.append(PageBreak())

    # ----- 3. Preprocessing Pipeline -----
    story.append(Paragraph("3. Preprocessing and Data Augmentation", heading2_style))
    story.append(Paragraph(
        "The integrity of a deep learning model is intrinsically tied to data quality. The ALS project implements a rigorous pre-processing pipeline:",
        body_style
    ))
    
    pp_points = [
        "CLAHE (Contrast Limited Adaptive Histogram Equalization): Applied to amplify the contrast of connective tissues, making fibrotic septa distinctly visible against healthy hepatocytes.",
        "Standardization: Images are resized to 224x224 (or 384x384 for high-res inputs) and normalized using standard ImageNet mean and variance to align with pre-trained initialization states.",
        "Augmentation: During training, models are subjected to rotations, color jitter, and horizontal flips to achieve rotational invariance."
    ]
    for pt in pp_points:
        story.append(Paragraph(f"&bull; {pt}", bullet_style))

    # ----- 4. Production API & Application -----
    story.append(Paragraph("4. Production Interfaces", heading2_style))
    story.append(Paragraph(
        "The project is packaged into two distinct deployment interfaces for medical professionals:",
        body_style
    ))
    story.append(Paragraph("Live Streamlit Dashboard", heading3_style))
    story.append(Paragraph(
        "An interactive, user-friendly frontend built in Streamlit. It allows clinicians to drag-and-drop biopsy scans, preview CLAHE enhancements in real-time, "
        "and select between the specific individual models or the combined Ensemble framework. "
        "It outputs a high-level visual confidence distribution chart calculated via the softmax outputs.",
        body_style
    ))
    story.append(Paragraph("CLI Reporting Suite", heading3_style))
    story.append(Paragraph(
        "A robust suite of python scripts capable of generating automated Grad-CAM heatmaps, batch predictions, and comparative ROC curve analysis. "
        "This allows for offline, programmatic evaluation of thousands of records simultaneously.",
        body_style
    ))

    # ----- 5. Conclusion -----
    story.append(Spacer(1, 20))
    story.append(Paragraph("5. Future Scope", heading2_style))
    story.append(Paragraph(
        "The integration of the 4-model ensemble has stabilized the cross-validation metrics immensely. "
        "Future iterations of the ALS project will look toward directly interfacing the API with hospital Electronic Health Record (EHR) systems and adding Explainable AI (XAI) overlays directly into the UI to pinpoint exactly which pixels influenced the F-stage classification.",
        body_style
    ))

    # Build the PDF
    doc.build(story)
    print(f"Success! Detailed report generated at: {filename}")

if __name__ == "__main__":
    build_pdf()
