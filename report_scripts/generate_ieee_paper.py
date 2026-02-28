<<<<<<< HEAD

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import re
from pathlib import Path

# --- Configuration ---
MD_FILE = Path(r"C:\Users\range\.gemini\antigravity\brain\57df14e0-8d6c-457d-8f54-762e47ea3c1f\conference_paper_draft.md")
PDF_OUTPUT = Path(r"d:\ALS\Research_Materials\PDFs\Reports\IEEE_Conference_Paper.pdf")
IMAGE_PATH = Path(r"d:\ALS\Research_Materials\PDFs\Plots_and_Figures\ensemble_confusion_matrix.png")

# Layout Constants (points)
PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11.0
MARGIN_X = 0.75
MARGIN_Y = 0.75
COL_GAP = 0.3
COL_WIDTH = (PAGE_WIDTH - 2*MARGIN_X - COL_GAP) / 2
FONT_FAMILY = 'serif' # Times New Roman approx

# Text Styles
STYLES = {
    'title': {'fontsize': 16, 'fontweight': 'bold', 'ha': 'center'},
    'author': {'fontsize': 11, 'fontstyle': 'italic', 'ha': 'center'},
    'abstract': {'fontsize': 9, 'fontweight': 'bold', 'style': 'italic'},
    'h1': {'fontsize': 10, 'fontweight': 'bold', 'ha': 'center', 'upper': True}, # I. INTRODUCTION
    'h2': {'fontsize': 10, 'fontstyle': 'italic', 'ha': 'left'},
    'body': {'fontsize': 9, 'ha': 'justify'},
    'caption': {'fontsize': 8, 'ha': 'center', 'style': 'italic'}
}

class PDFGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.pdf = PdfPages(filename)
        self.fig = None
        self.y_cursor = 0
        self.col_idx = 0 # 0 or 1
        self.x_cursors = [MARGIN_X, MARGIN_X + COL_WIDTH + COL_GAP]
        self.current_page = 0
        self.new_page()

    def new_page(self):
        if self.fig:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        
        self.fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        self.fig.patch.set_facecolor('white')
        self.y_cursor = PAGE_HEIGHT - MARGIN_Y
        self.col_idx = 0
        self.current_page += 1
        # Debugging grid
        # plt.axvline(self.x_cursors[0], color='r', alpha=0.1)
        # plt.axvline(self.x_cursors[0]+COL_WIDTH, color='r', alpha=0.1)
        # plt.axvline(self.x_cursors[1], color='b', alpha=0.1)
        # plt.axvline(self.x_cursors[1]+COL_WIDTH, color='b', alpha=0.1)
        plt.axis('off')

    def check_space(self, required_height):
        # If not enough space, move to next column/page
        if self.y_cursor - required_height < MARGIN_Y:
            if self.col_idx == 0:
                self.col_idx = 1
                self.y_cursor = PAGE_HEIGHT - MARGIN_Y
            else:
                self.new_page()

    def render_title_block(self, lines):
        # Renders across full page width
        x_center = PAGE_WIDTH / 2
        
        # Title
        title = lines[0].replace('# ', '')
        wrapped_title = textwrap.wrap(title, width=60)
        for line in wrapped_title:
            self.fig.text(x_center, self.y_cursor/PAGE_HEIGHT, line, **STYLES['title'])
            self.y_cursor -= 0.35
        
        self.y_cursor -= 0.2
        
        # Author Block (next few lines until ---)
        for line in lines[2:]:
            if '---' in line: break
            if not line.strip(): continue
            self.fig.text(x_center, self.y_cursor/PAGE_HEIGHT, line.strip('*'), **STYLES['author'])
            self.y_cursor -= 0.2
            
        self.y_cursor -= 0.2
        # Reset to 2-column mode starts from this Y
        self.start_y_2col = self.y_cursor

    def render_text_block(self, text, style_key, indent=0.0):
        style = STYLES[style_key]
        
        # Estimate wrap width based on font size (heuristic)
        cutoff = 55 if style_key == 'body' else 45
        if style_key == 'abstract': cutoff = 50
        
        wrapper = textwrap.TextWrapper(width=cutoff)
        lines = wrapper.wrap(text)
        
        line_height = style['fontsize'] * 1.5 / 72.0 # pts to inches
        
        # Check total block size, might break paragraphs but keeping simple
        # If header, keep with next line -> strict check
        
        for line in lines:
            self.check_space(line_height)
            
            x_pos = self.x_cursors[self.col_idx] + indent
            y_pos_norm = self.y_cursor / PAGE_HEIGHT
            
            # Matplotlib text coords are 0-1
            fw = style.get('fontweight', 'normal')
            fs = style.get('fontstyle', 'normal')
            if style_key == 'h1' and style.get('upper'): line = line.upper()
            
            self.fig.text(x_pos/PAGE_WIDTH, y_pos_norm, line, 
                          fontsize=style['fontsize'], fontweight=fw, fontstyle=fs, 
                          ha='left', fontfamily=FONT_FAMILY) # Force left align for flow
            
            self.y_cursor -= line_height
            
        self.y_cursor -= line_height * 0.5 # Paragraph spacing

    def render_image(self):
        if not IMAGE_PATH.exists(): return
        
        # Image takes about 3 inches height
        h = 3.0
        self.check_space(h)
        
        img = plt.imread(str(IMAGE_PATH))
        
        # Calculate bounding box in 0-1 coords
        # Left, Bottom, Width, Height
        x0 = self.x_cursors[self.col_idx] / PAGE_WIDTH
        w = COL_WIDTH / PAGE_WIDTH
        y0 = (self.y_cursor - h) / PAGE_HEIGHT
        h_norm = h / PAGE_HEIGHT
        
        if y0 < 0: # Should be caught by check_space but safety
            self.new_page()
            y0 = (PAGE_HEIGHT - MARGIN_Y - h) / PAGE_HEIGHT
            
        ax = self.fig.add_axes([x0, y0, w, h_norm])
        ax.imshow(img)
        ax.axis('off')
        
        self.y_cursor -= h
        self.fig.text((x0 + w/2), (y0 - 0.02), "Fig. 1. Confusion Matrix", **STYLES['caption'])
        self.y_cursor -= 0.3

    def save(self):
        self.pdf.savefig(self.fig)
        self.pdf.close()

def parse_markdown_and_render():
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    
    # Init PDF
    pdf = PDFGenerator(PDF_OUTPUT)
    
    # 1. Render Title Block (Hardcoded extraction for first few lines)
    pdf.render_title_block(lines[:10])
    
    # 2. Parse rest
    body_lines = lines[15:] # Skip title/author block
    
    current_text = []
    
    for line in body_lines:
        line = line.strip()
        if not line: continue
        
        # Check headers
        if line.startswith('## '):
            # Flush prev
            if current_text: pdf.render_text_block(" ".join(current_text), 'body')
            current_text = []
            
            # H1
            header = line.replace('## ', '')
            pdf.render_text_block(header, 'h1')
            
        elif line.startswith('### '):
            if current_text: pdf.render_text_block(" ".join(current_text), 'body')
            current_text = []
             # H2
            header = line.replace('### ', '')
            pdf.render_text_block(header, 'h2')
            
        elif line.startswith('**_Abstract_'):
            txt = line.replace('**_Abstract_—', '').replace('**', '')
            pdf.render_text_block("ABSTRACT", 'h1')
            pdf.render_text_block(txt, 'abstract')
            
        elif line.startswith('**_Keywords'):
             txt = line.replace('**_Keywords—', '').replace('**', '').replace('_', '')
             pdf.render_text_block("Keywords: " + txt, 'abstract')
             pdf.y_cursor -= 0.2

        elif line.startswith('_(Figure 1'):
            if current_text: pdf.render_text_block(" ".join(current_text), 'body')
            current_text = []
            pdf.render_image() # Insert image
            
        elif line.startswith('|'):
             # Skip tables for now in this simple renderer, just add note
             if "Accuracy" in line:
                 pdf.render_text_block("[Table 1: Comparative Evaluation Metrics Omitted - See Source]", 'caption')
             continue
             
        elif line.startswith('$$'):
             # Math formula
             pdf.render_text_block("Pc = Sum(w * p) / Sum(w)", 'body', indent=0.5)
             
        else:
            # Body text
            current_text.append(line)
            
    # Flush last
    if current_text: pdf.render_text_block(" ".join(current_text), 'body')
    
    pdf.save()
    print(f"PDF Generated: {PDF_OUTPUT}")

if __name__ == "__main__":
    parse_markdown_and_render()
=======

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import re
from pathlib import Path

# --- Configuration ---
MD_FILE = Path(r"C:\Users\range\.gemini\antigravity\brain\57df14e0-8d6c-457d-8f54-762e47ea3c1f\conference_paper_draft.md")
PDF_OUTPUT = Path(r"d:\ALS\Research_Materials\PDFs\Reports\IEEE_Conference_Paper.pdf")
IMAGE_PATH = Path(r"d:\ALS\Research_Materials\PDFs\Plots_and_Figures\ensemble_confusion_matrix.png")

# Layout Constants (points)
PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11.0
MARGIN_X = 0.75
MARGIN_Y = 0.75
COL_GAP = 0.3
COL_WIDTH = (PAGE_WIDTH - 2*MARGIN_X - COL_GAP) / 2
FONT_FAMILY = 'serif' # Times New Roman approx

# Text Styles
STYLES = {
    'title': {'fontsize': 16, 'fontweight': 'bold', 'ha': 'center'},
    'author': {'fontsize': 11, 'fontstyle': 'italic', 'ha': 'center'},
    'abstract': {'fontsize': 9, 'fontweight': 'bold', 'style': 'italic'},
    'h1': {'fontsize': 10, 'fontweight': 'bold', 'ha': 'center', 'upper': True}, # I. INTRODUCTION
    'h2': {'fontsize': 10, 'fontstyle': 'italic', 'ha': 'left'},
    'body': {'fontsize': 9, 'ha': 'justify'},
    'caption': {'fontsize': 8, 'ha': 'center', 'style': 'italic'}
}

class PDFGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.pdf = PdfPages(filename)
        self.fig = None
        self.y_cursor = 0
        self.col_idx = 0 # 0 or 1
        self.x_cursors = [MARGIN_X, MARGIN_X + COL_WIDTH + COL_GAP]
        self.current_page = 0
        self.new_page()

    def new_page(self):
        if self.fig:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        
        self.fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        self.fig.patch.set_facecolor('white')
        self.y_cursor = PAGE_HEIGHT - MARGIN_Y
        self.col_idx = 0
        self.current_page += 1
        # Debugging grid
        # plt.axvline(self.x_cursors[0], color='r', alpha=0.1)
        # plt.axvline(self.x_cursors[0]+COL_WIDTH, color='r', alpha=0.1)
        # plt.axvline(self.x_cursors[1], color='b', alpha=0.1)
        # plt.axvline(self.x_cursors[1]+COL_WIDTH, color='b', alpha=0.1)
        plt.axis('off')

    def check_space(self, required_height):
        # If not enough space, move to next column/page
        if self.y_cursor - required_height < MARGIN_Y:
            if self.col_idx == 0:
                self.col_idx = 1
                self.y_cursor = PAGE_HEIGHT - MARGIN_Y
            else:
                self.new_page()

    def render_title_block(self, lines):
        # Renders across full page width
        x_center = PAGE_WIDTH / 2
        
        # Title
        title = lines[0].replace('# ', '')
        wrapped_title = textwrap.wrap(title, width=60)
        for line in wrapped_title:
            self.fig.text(x_center, self.y_cursor/PAGE_HEIGHT, line, **STYLES['title'])
            self.y_cursor -= 0.35
        
        self.y_cursor -= 0.2
        
        # Author Block (next few lines until ---)
        for line in lines[2:]:
            if '---' in line: break
            if not line.strip(): continue
            self.fig.text(x_center, self.y_cursor/PAGE_HEIGHT, line.strip('*'), **STYLES['author'])
            self.y_cursor -= 0.2
            
        self.y_cursor -= 0.2
        # Reset to 2-column mode starts from this Y
        self.start_y_2col = self.y_cursor

    def render_text_block(self, text, style_key, indent=0.0):
        style = STYLES[style_key]
        
        # Estimate wrap width based on font size (heuristic)
        cutoff = 55 if style_key == 'body' else 45
        if style_key == 'abstract': cutoff = 50
        
        wrapper = textwrap.TextWrapper(width=cutoff)
        lines = wrapper.wrap(text)
        
        line_height = style['fontsize'] * 1.5 / 72.0 # pts to inches
        
        # Check total block size, might break paragraphs but keeping simple
        # If header, keep with next line -> strict check
        
        for line in lines:
            self.check_space(line_height)
            
            x_pos = self.x_cursors[self.col_idx] + indent
            y_pos_norm = self.y_cursor / PAGE_HEIGHT
            
            # Matplotlib text coords are 0-1
            fw = style.get('fontweight', 'normal')
            fs = style.get('fontstyle', 'normal')
            if style_key == 'h1' and style.get('upper'): line = line.upper()
            
            self.fig.text(x_pos/PAGE_WIDTH, y_pos_norm, line, 
                          fontsize=style['fontsize'], fontweight=fw, fontstyle=fs, 
                          ha='left', fontfamily=FONT_FAMILY) # Force left align for flow
            
            self.y_cursor -= line_height
            
        self.y_cursor -= line_height * 0.5 # Paragraph spacing

    def render_image(self):
        if not IMAGE_PATH.exists(): return
        
        # Image takes about 3 inches height
        h = 3.0
        self.check_space(h)
        
        img = plt.imread(str(IMAGE_PATH))
        
        # Calculate bounding box in 0-1 coords
        # Left, Bottom, Width, Height
        x0 = self.x_cursors[self.col_idx] / PAGE_WIDTH
        w = COL_WIDTH / PAGE_WIDTH
        y0 = (self.y_cursor - h) / PAGE_HEIGHT
        h_norm = h / PAGE_HEIGHT
        
        if y0 < 0: # Should be caught by check_space but safety
            self.new_page()
            y0 = (PAGE_HEIGHT - MARGIN_Y - h) / PAGE_HEIGHT
            
        ax = self.fig.add_axes([x0, y0, w, h_norm])
        ax.imshow(img)
        ax.axis('off')
        
        self.y_cursor -= h
        self.fig.text((x0 + w/2), (y0 - 0.02), "Fig. 1. Confusion Matrix", **STYLES['caption'])
        self.y_cursor -= 0.3

    def save(self):
        self.pdf.savefig(self.fig)
        self.pdf.close()

def parse_markdown_and_render():
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    
    # Init PDF
    pdf = PDFGenerator(PDF_OUTPUT)
    
    # 1. Render Title Block (Hardcoded extraction for first few lines)
    pdf.render_title_block(lines[:10])
    
    # 2. Parse rest
    body_lines = lines[15:] # Skip title/author block
    
    current_text = []
    
    for line in body_lines:
        line = line.strip()
        if not line: continue
        
        # Check headers
        if line.startswith('## '):
            # Flush prev
            if current_text: pdf.render_text_block(" ".join(current_text), 'body')
            current_text = []
            
            # H1
            header = line.replace('## ', '')
            pdf.render_text_block(header, 'h1')
            
        elif line.startswith('### '):
            if current_text: pdf.render_text_block(" ".join(current_text), 'body')
            current_text = []
             # H2
            header = line.replace('### ', '')
            pdf.render_text_block(header, 'h2')
            
        elif line.startswith('**_Abstract_'):
            txt = line.replace('**_Abstract_—', '').replace('**', '')
            pdf.render_text_block("ABSTRACT", 'h1')
            pdf.render_text_block(txt, 'abstract')
            
        elif line.startswith('**_Keywords'):
             txt = line.replace('**_Keywords—', '').replace('**', '').replace('_', '')
             pdf.render_text_block("Keywords: " + txt, 'abstract')
             pdf.y_cursor -= 0.2

        elif line.startswith('_(Figure 1'):
            if current_text: pdf.render_text_block(" ".join(current_text), 'body')
            current_text = []
            pdf.render_image() # Insert image
            
        elif line.startswith('|'):
             # Skip tables for now in this simple renderer, just add note
             if "Accuracy" in line:
                 pdf.render_text_block("[Table 1: Comparative Evaluation Metrics Omitted - See Source]", 'caption')
             continue
             
        elif line.startswith('$$'):
             # Math formula
             pdf.render_text_block("Pc = Sum(w * p) / Sum(w)", 'body', indent=0.5)
             
        else:
            # Body text
            current_text.append(line)
            
    # Flush last
    if current_text: pdf.render_text_block(" ".join(current_text), 'body')
    
    pdf.save()
    print(f"PDF Generated: {PDF_OUTPUT}")

if __name__ == "__main__":
    parse_markdown_and_render()
>>>>>>> origin/main
