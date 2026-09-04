import re

# Sample text
text = """
Hello, this is a sample text.

Visit https://www.example.com for more information.
You can email us at example@email.com.

The date is 09/04/2026 or 04-09-2026.
Call us at 123-456-7890.
Our ZIP code is 12345 or 12345-6789.

This is a test @#$%^&* with punctuation!
The application is approved. #example #Python

    There     are      extra       spaces.

Numbers such as 123 and 456 should be removed.
"""

# --------------------------------------------------
# 1. Removing HTML Tags
# --------------------------------------------------

html_text = "<p>Hello <b>World</b>!</p>"

cleaned_html = re.sub(r'<.*?>', '', html_text)

print("1. Removing HTML Tags:")
print(cleaned_html)


# --------------------------------------------------
# 2. Removing Non-Alphanumeric Characters
# --------------------------------------------------

cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

print("\n2. Removing Non-Alphanumeric Characters:")
print(cleaned_text)


# --------------------------------------------------
# 3. Removing Extra Whitespace
# --------------------------------------------------

cleaned_text = re.sub(r'\s+', ' ', text)

print("\n3. Removing Extra Whitespace:")
print(cleaned_text)


# --------------------------------------------------
# 4. Extracting Email Addresses
# --------------------------------------------------

emails = re.findall(r'\S+@\S+', text)

print("\n4. Extracting Email Addresses:")
print(emails)


# --------------------------------------------------
# 5. Extracting URLs
# --------------------------------------------------

urls = re.findall(r'https?://\S+', text)

print("\n5. Extracting URLs:")
print(urls)


# --------------------------------------------------
# 6. Extracting Dates
# Example: MM/DD/YYYY or DD-MM-YYYY
# --------------------------------------------------

dates = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', text)

print("\n6. Extracting Dates:")
print(dates)


# --------------------------------------------------
# 7. Extracting Phone Numbers
# Example: XXX-XXX-XXXX
# --------------------------------------------------

phone_numbers = re.findall(r'\d{3}-\d{3}-\d{4}', text)

print("\n7. Extracting Phone Numbers:")
print(phone_numbers)


# --------------------------------------------------
# 8. Extracting ZIP Codes
# Example: XXXXX or XXXXX-XXXX
# --------------------------------------------------

zip_codes = re.findall(r'\b\d{5}(?:-\d{4})?\b', text)

print("\n8. Extracting ZIP Codes:")
print(zip_codes)


# --------------------------------------------------
# 9. Removing Punctuation
# --------------------------------------------------

cleaned_text = re.sub(r'[^\w\s]', '', text)

print("\n9. Removing Punctuation:")
print(cleaned_text)


# --------------------------------------------------
# 10. Replacing Special Characters
# --------------------------------------------------

cleaned_text = re.sub(r'[@#$%^&*]', '', text)

print("\n10. Replacing Special Characters:")
print(cleaned_text)


# --------------------------------------------------
# 11. Cleaning Whitespaces at Beginning and End
# --------------------------------------------------

cleaned_text = text.strip()

print("\n11. Cleaning Beginning and Ending Whitespace:")
print(cleaned_text)


# --------------------------------------------------
# 12. Matching Words Starting with "app"
# --------------------------------------------------

matches = re.findall(r'\bapp\w*\b', text, re.IGNORECASE)

print("\n12. Words Starting with 'app':")
print(matches)


# --------------------------------------------------
# 13. Replacing Multiple Spaces with Single Space
# --------------------------------------------------

cleaned_text = re.sub(r'\s+', ' ', text)

print("\n13. Replacing Multiple Spaces:")
print(cleaned_text)


# --------------------------------------------------
# 14. Removing Numbers
# --------------------------------------------------

cleaned_text = re.sub(r'\d+', '', text)

print("\n14. Removing Numbers:")
print(cleaned_text)


# --------------------------------------------------
# 15. Extracting Hashtags
# --------------------------------------------------

hashtags = re.findall(r'#\w+', text)

print("\n15. Extracting Hashtags:")
print(hashtags)