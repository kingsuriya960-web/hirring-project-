
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_sample_pdf(filename="sample_enterprise_agreement.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 750, "ENTERPRISE CLOUD SERVICES AGREEMENT")
    
    c.setFont("Helvetica", 10)
    lines = [
        "This Enterprise Cloud Services Agreement ('Agreement') is made effective as of November 1, 2026.",
        "",
        "SECTION 1: TERM AND TERMINATION",
        "1.1 Termination for Convenience: Either party may terminate this Agreement without cause upon providing",
        "    at least thirty (30) days prior written notice to the other party.",
        "1.2 Termination for Cause: In the event of material breach, the non-breaching party may terminate this",
        "    Agreement if such breach remains uncured for a period of fifteen (15) days following written notice.",
        "",
        "SECTION 2: DATA PROTECTION AND SECURITY",
        "2.1 Encryption Standards: Vendor warrants that all Customer Data shall be encrypted using AES-256 at rest",
        "    and TLS 1.3 in transit across all public and private networks.",
        "2.2 Security Incident Notification: Vendor shall formally notify Customer in writing within seventy-two (72)",
        "    hours of confirming any security incident or unauthorized disclosure of Customer Data.",
        "",
        "SECTION 3: LIMITATION OF LIABILITY",
        "3.1 Liability Ceiling: In no event shall either party's aggregate liability exceed the total fees paid or payable",
        "    by Customer in the preceding twelve (12) months.",
        "3.2 Waiver of Consequential Damages: Neither party shall be liable for indirect, incidental, punitive, or",
        "    consequential damages under any legal theory."
    ]
    
    y = 710
    for line in lines:
        if line.startswith("SECTION"):
            c.setFont("Helvetica-Bold", 11)
        elif line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFont("Helvetica", 10)
        c.drawString(72, y, line)
        y -= 18
        
    c.showPage()
    c.save()
    print(f"Created {filename} successfully.")

if __name__ == '__main__':
    create_sample_pdf()

