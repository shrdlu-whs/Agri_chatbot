from PyPDF2 import PdfReader, PdfWriter

# Define your new metadata keys and values
new_metadata = {
    "/source": "European Commission",
    "/creator": "European Commission",
    "/producer": "Acrobat",
}

# Paths to input and output PDF files
file_name = "direct-payments-eligibility-conditions_en"
input_pdf_path = f"{file_name}.pdf"
output_pdf_path = f"{file_name}_w_metadata.pdf"

# Create a PdfReader and PdfWriter instance
reader = PdfReader(input_pdf_path)
writer = PdfWriter()

# Copy pages from the input PDF to the writer
for page in reader.pages:
    writer.add_page(page)

# Extract existing metadata from the PDF
existing_metadata = reader.metadata
if existing_metadata:
    print("Existing Metadata:", existing_metadata)

# Merge existing metadata with the new metadata
updated_metadata = existing_metadata.copy() if existing_metadata else {}
updated_metadata.update(new_metadata)

# Add the merged metadata to the writer
writer.add_metadata(updated_metadata)

# Write the output PDF with the updated metadata
with open(output_pdf_path, "wb") as output_pdf:
    writer.write(output_pdf)

print(f"Metadata updated and saved to {output_pdf_path}")
