"""
Generate Research Paper Abstract PDF for Liver Fibrosis Staging Project
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from pathlib import Path

# Output path
OUTPUT_PATH = Path(__file__).parent / "research_paper_abstract.pdf"

def create_abstract_pdf():
    """Create a professional PDF with the research paper abstract."""
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=1*inch,
        leftMargin=1*inch,
        topMargin=1*inch,
        bottomMargin=1*inch
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=HexColor('#1a1a2e'),
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=30,
        textColor=HexColor('#4a4a4a'),
        fontName='Helvetica-Oblique'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=HexColor('#16213e'),
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        textColor=HexColor('#2d2d2d'),
        firstLineIndent=20
    )
    
    keywords_style = ParagraphStyle(
        'Keywords',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceBefore=20,
        textColor=HexColor('#333333'),
        fontName='Helvetica'
    )
    
    # Content
    story = []
    
    # Title
    title = "Explainable Vision Transformer for Automated Liver Fibrosis Staging: A Deep Learning Approach with Attention Visualization"
    story.append(Paragraph(title, title_style))
    
    # Author placeholder
    story.append(Paragraph("Research Paper Abstract", subtitle_style))
    
    # Abstract heading
    story.append(Paragraph("ABSTRACT", heading_style))
    
    # Abstract paragraphs
    abstract_p1 = """
    Liver fibrosis is a progressive condition characterized by excessive accumulation of 
    extracellular matrix proteins, primarily resulting from chronic liver diseases such as 
    hepatitis B and C, alcoholic liver disease, and non-alcoholic steatohepatitis (NASH). 
    Accurate staging of liver fibrosis (F0-F4) is crucial for clinical decision-making, 
    treatment planning, and prognosis assessment. Traditional methods rely on invasive 
    liver biopsies preceded by histopathological analysis, which are subject to sampling 
    variability and inter-observer disagreement. This study presents an automated, 
    non-invasive deep learning framework for liver fibrosis staging using histopathological 
    images with explainable artificial intelligence (XAI) capabilities.
    """
    story.append(Paragraph(abstract_p1, body_style))
    
    abstract_p2 = """
    We propose a Vision Transformer (ViT-B-16) based architecture with pre-trained 
    ImageNet weights, fine-tuned for five-class liver fibrosis classification (F0: No 
    fibrosis, F1: Portal fibrosis, F2: Periportal fibrosis, F3: Bridging fibrosis, F4: 
    Cirrhosis). The input images undergo Contrast Limited Adaptive Histogram Equalization 
    (CLAHE) preprocessing to enhance local contrast and highlight fibrotic tissue patterns. 
    The model employs dropout regularization (p=0.3) to prevent overfitting and utilizes 
    AdamW optimizer with label smoothing cross-entropy loss for robust training. Early 
    stopping with patience monitoring ensures optimal model convergence without 
    overtraining.
    """
    story.append(Paragraph(abstract_p2, body_style))
    
    abstract_p3 = """
    To address the critical need for interpretability in medical AI systems, we integrate 
    Gradient-weighted Class Activation Mapping (Grad-CAM) visualization to generate 
    attention heatmaps highlighting regions of the input image most influential for the 
    model's predictions. This explainability component enables clinicians to validate 
    model decisions and builds trust in automated diagnostic systems. The attention 
    visualizations demonstrate that the model correctly focuses on fibrotic tissue 
    patterns, portal areas, and architectural distortions characteristic of different 
    fibrosis stages.
    """
    story.append(Paragraph(abstract_p3, body_style))
    
    abstract_p4 = """
    The model was validated using stratified train-validation splits to ensure balanced 
    class representation. Performance evaluation includes multi-class accuracy, weighted 
    F1-score, confusion matrix analysis, and Cohen's Kappa score with quadratic weighting 
    to account for the ordinal nature of fibrosis staging. The quadratic-weighted Kappa 
    is particularly appropriate as misclassifications between adjacent stages (e.g., F1 
    vs F2) are penalized less severely than distant misclassifications (e.g., F0 vs F4), 
    reflecting clinical significance. K-fold cross-validation with 95% confidence 
    intervals ensures statistical robustness of reported metrics.
    """
    story.append(Paragraph(abstract_p4, body_style))
    
    abstract_p5 = """
    The proposed framework demonstrates the potential of transformer-based architectures 
    for automated liver fibrosis staging while maintaining clinical interpretability 
    through explainable AI techniques. This approach offers a scalable, reproducible, 
    and objective alternative to traditional histopathological assessment, potentially 
    reducing diagnostic variability and improving patient outcomes through early and 
    accurate fibrosis detection.
    """
    story.append(Paragraph(abstract_p5, body_style))
    
    # Keywords
    keywords_text = """
    <b>Keywords:</b> Liver Fibrosis Staging, Vision Transformer (ViT), Deep Learning, 
    Explainable AI, Grad-CAM, Medical Image Classification, CLAHE Preprocessing, 
    Cohen's Kappa, Histopathology Analysis, Computer-Aided Diagnosis
    """
    story.append(Paragraph(keywords_text, keywords_style))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Abstract PDF generated successfully!")
    print(f"📄 Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_abstract_pdf()
