import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime

def generate_pdf_report(dataframe, filename="road_damage_report.pdf"):
    """
    Generates a beautifully formatted PDF report of detected road damage.
    """
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles matching our primary brand theme (Deep Navy Blue)
    primary_color = colors.HexColor("#1E3A8A")
    text_color = colors.HexColor("#1E293B")
    
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=primary_color,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=text_color,
        spaceAfter=25
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=15,
        spaceAfter=10
    )
    
    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=text_color
    )
    
    header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.white
    )

    # Title & Metadata
    story.append(Paragraph("AI-Powered Road Damage Inspection Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
                           f"Total Incidents Logged: {len(dataframe)}<br/>"
                           f"Jurisdiction: National Highway Authority (NHA) & Municipal Corporations", subtitle_style))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Damage Severity Summary Table", section_style))
    
    # Summary of severities
    low_count = sum(dataframe['severity'] == 'Low')
    med_count = sum(dataframe['severity'] == 'Medium')
    high_count = sum(dataframe['severity'] == 'High')
    
    summary_data = [
        [Paragraph("Severity Level", header_style), Paragraph("Count", header_style), Paragraph("Action Recommended", header_style)],
        [Paragraph("High", cell_style), Paragraph(str(high_count), cell_style), Paragraph("Immediate repair required (within 24-48 hours)", cell_style)],
        [Paragraph("Medium", cell_style), Paragraph(str(med_count), cell_style), Paragraph("Schedule for routine maintenance (within 2 weeks)", cell_style)],
        [Paragraph("Low", cell_style), Paragraph(str(low_count), cell_style), Paragraph("Monitor; include in periodic resurfacing plans", cell_style)],
    ]
    
    summary_table = Table(summary_data, colWidths=[120, 80, 320])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])
    ]))
    story.append(summary_table)
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("Detailed Incident Log", section_style))
    
    # Detailed log columns
    log_headers = ["Timestamp", "Type", "Severity", "Conf.", "Road & Location", "GPS Coordinates"]
    log_data = [[Paragraph(h, header_style) for h in log_headers]]
    
    # Limit table length to top 25 recent items to keep PDF size / generation optimal
    df_limited = dataframe.head(25)
    
    for idx, row in df_limited.iterrows():
        gps = f"{row['latitude']:.4f}, {row['longitude']:.4f}"
        ts = row['timestamp']
        if len(ts) > 10:
            ts = ts[:10]  # Show date only or short timestamp to fit column
            
        log_data.append([
            Paragraph(ts, cell_style),
            Paragraph(row['damage_type'], cell_style),
            Paragraph(row['severity'], cell_style),
            Paragraph(f"{int(row['confidence']*100)}%", cell_style),
            Paragraph(row['road_name'], cell_style),
            Paragraph(gps, cell_style)
        ])
        
    log_table = Table(log_data, colWidths=[70, 110, 60, 40, 150, 90])
    log_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])
    ]))
    
    story.append(log_table)
    
    # Build Document
    doc.build(story)
    return filename
