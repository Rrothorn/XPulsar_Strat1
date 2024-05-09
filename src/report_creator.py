# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:58:37 2024

@author: Gebruiker
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
from datetime import datetime
from io import BytesIO


# Function to convert DataFrame to a list of lists
def df_to_list_of_lists(df):
    return [df.columns.tolist()] + df.values.tolist()


def generate_report(df, date):

    dfr = df[df.index == date]

    cols = ['ticker', 'buysell', 'betsize_cl', 'trade_open', 'trade_close', 'pnl_cl']

    dfr = dfr[cols][dfr.pnl_cl != 0]

    total_weight = (dfr['betsize_cl'].sum()* 100).round(3).astype(str) + '%'
    total_pnl_cl = (dfr['pnl_cl'].sum() * 100).round(3).astype(str) + '%'
    dfr['pnl_cl'] = (dfr['pnl_cl'] * 100).round(3).astype(str) + '%'  # Convert pnl_cl to percentage and round to 2 decimal places
    dfr = dfr.rename(columns={'pnl_cl':'PnL', 'betsize_cl':'weight'})

    # Convert DataFrame to a list of lists
    data = df_to_list_of_lists(dfr)

    # Add Total row
    total_row = ['Total', '', total_weight, '', '', total_pnl_cl]
    data.append(total_row)

    # Create PDF
    buffer = BytesIO()
    pdf_filename = str(date)+'daily_report.pdf'
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []

    # Add logo
    logo_path = '../assets/xpulsar-logo.png'  # Provide the path to your logo file
    logo = Image(logo_path, width=100, height=100)
    logo.drawWidth = 160
    logo.drawHeight = 120
    logo.hAlign = 'RIGHT'
    elements.append(logo)

    # Add text and date
    styleSheet = getSampleStyleSheet()
    text = f"Daily report for {date}"
    elements.append(Paragraph(text, styleSheet['BodyText']))

    # Add table
    table = Table(data)

    # Add style to the table
    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#1e3143'),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)])
    table.setStyle(style)
    
    # Add different style for the Total row
    total_style = TableStyle([('BACKGROUND', (0, -1), (-1, -1), '#672967'),
                              ('TEXTCOLOR', (0, -1), (-1, -1), colors.white),
                              ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')])
    table.setStyle(total_style)
    
    elements.append(table)
    
    #build pdf in memory
    doc.build(elements)
    return buffer.getvalue()



