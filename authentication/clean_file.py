# clean_file.py
input_path = "models.py"
output_path = "models_clean.py"

with open(input_path, "rb") as infile:
    content = infile.read()

# Remove all null bytes
clean_content = content.replace(b'\x00', b'')

with open(output_path, "wb") as outfile:
    outfile.write(clean_content)

print("File cleaned. New file created as:", output_path)
