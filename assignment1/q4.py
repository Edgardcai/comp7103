"""
COMP7103 Assignment 1 - Question 4
Classification in Weka: Gallstone Dataset Processing and ARFF File Generation
"""

import pandas as pd
import os
import zipfile
import xml.etree.ElementTree as ET

def read_excel_manually(excel_file):
    """
    Manually parse Excel file from ZIP structure.
    This is needed because the Excel file has format issues with standard readers.
    """
    with zipfile.ZipFile(excel_file, 'r') as z:
        # Read shared strings
        shared_strings = []
        ss_xml = z.read('xl/sharedStrings.xml')
        ss_root = ET.fromstring(ss_xml)
        for si in ss_root:
            for t in si:
                if t.tag.endswith('t'):
                    shared_strings.append(t.text)

        # Read sheet data
        sheet_xml = z.read('xl/worksheets/sheet1.xml')
        sheet_root = ET.fromstring(sheet_xml)

        # Extract all rows
        data = []
        for row_elem in sheet_root.iter():
            if row_elem.tag.endswith('row'):
                row_data = []
                for cell in row_elem:
                    if cell.tag.endswith('c'):
                        cell_type = cell.get('t', 'n')
                        v_elem = cell.find('.//{http://purl.oclc.org/ooxml/spreadsheetml/main}v')
                        if v_elem is not None:
                            value = v_elem.text
                            if cell_type == 's':
                                row_data.append(shared_strings[int(value)])
                            else:
                                try:
                                    if '.' in value:
                                        row_data.append(float(value))
                                    else:
                                        row_data.append(int(value))
                                except:
                                    row_data.append(value)
                        else:
                            row_data.append(None)
                if row_data:
                    data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

def preprocess_gallstone_data(excel_file='gallstone.xlsx'):
    """
    Pre-process the Gallstone dataset by extracting and transforming required attributes.

    Args:
        excel_file: Path to the gallstone.xlsx file

    Returns:
        DataFrame with preprocessed data
    """
    # Read the Excel file manually (due to format issues)
    df = read_excel_manually(excel_file)

    # Create new dataframe with transformed attributes
    processed_df = pd.DataFrame()

    # Gender: 0 = Male, 1 = Female
    processed_df['Gender'] = df['Gender'].map({0: 'Male', 1: 'Female'})

    # Comorbidity: 0 = Yes, all other values = No
    processed_df['Comorbidity'] = df['Comorbidity'].apply(lambda x: 'Yes' if x == 0 else 'No')

    # CAD (Coronary Artery Disease): 0 = No, 1 = Yes
    processed_df['CAD'] = df['Coronary Artery Disease (CAD)'].map({0: 'No', 1: 'Yes'})

    # Hypothyroidism: 0 = No, 1 = Yes
    processed_df['Hypothyroidism'] = df['Hypothyroidism'].map({0: 'No', 1: 'Yes'})

    # Hyperlipidemia: 0 = No, 1 = Yes
    processed_df['Hyperlipidemia'] = df['Hyperlipidemia'].map({0: 'No', 1: 'Yes'})

    # DM (Diabetes Mellitus): 0 = No, 1 = Yes
    processed_df['DM'] = df['Diabetes Mellitus (DM)'].map({0: 'No', 1: 'Yes'})

    # HFA (Hepatic Fat Accumulation): 0 = No, all other values = Yes
    processed_df['HFA'] = df['Hepatic Fat Accumulation (HFA)'].apply(lambda x: 'No' if x == 0 else 'Yes')

    # Class (Gallstone Status): 0 = No, 1 = Yes
    processed_df['Class'] = df['Gallstone Status'].map({0: 'No', 1: 'Yes'})

    return processed_df


def generate_arff_file(df, output_file='gallstone.arff', relation_name='gallstone'):
    """
    Generate an ARFF file from the preprocessed dataframe.

    Args:
        df: Preprocessed DataFrame
        output_file: Path to output ARFF file
        relation_name: Name of the relation in ARFF file
    """
    with open(output_file, 'w') as f:
        # Write relation name
        f.write(f"@RELATION {relation_name}\n\n")

        # Write attribute definitions
        f.write("@ATTRIBUTE Gender {Male,Female}\n")
        f.write("@ATTRIBUTE Comorbidity {Yes,No}\n")
        f.write("@ATTRIBUTE CAD {Yes,No}\n")
        f.write("@ATTRIBUTE Hypothyroidism {Yes,No}\n")
        f.write("@ATTRIBUTE Hyperlipidemia {Yes,No}\n")
        f.write("@ATTRIBUTE DM {Yes,No}\n")
        f.write("@ATTRIBUTE HFA {Yes,No}\n")
        f.write("@ATTRIBUTE Class {Yes,No}\n\n")

        # Write data section
        f.write("@DATA\n")

        # Write data instances
        for _, row in df.iterrows():
            # Handle missing values by replacing NaN with '?'
            values = []
            for val in row:
                if pd.isna(val):
                    values.append('?')
                else:
                    values.append(str(val))
            f.write(','.join(values) + '\n')

    print(f"ARFF file generated: {output_file}")


def display_arff_header():
    """
    Display the ARFF file header sections (before @DATA section) for Question 4a.
    """
    header = """@RELATION gallstone

@ATTRIBUTE Gender {Male,Female}
@ATTRIBUTE Comorbidity {Yes,No}
@ATTRIBUTE CAD {Yes,No}
@ATTRIBUTE Hypothyroidism {Yes,No}
@ATTRIBUTE Hyperlipidemia {Yes,No}
@ATTRIBUTE DM {Yes,No}
@ATTRIBUTE HFA {Yes,No}
@ATTRIBUTE Class {Yes,No}
"""
    print("=" * 60)
    print("ARFF File Header (for Question 4a)")
    print("=" * 60)
    print(header)
    print("=" * 60)
    return header


def main():
    """
    Main function to execute the preprocessing and ARFF generation.
    """
    print("COMP7103 Assignment 1 - Question 4")
    print("Processing Gallstone Dataset\n")

    # Check if the Excel file exists
    excel_file = 'gallstone.xlsx'
    if not os.path.exists(excel_file):
        print(f"Error: {excel_file} not found in current directory.")
        print(f"Current directory: {os.getcwd()}")
        print("Please ensure gallstone.xlsx is in the same directory as this script.")
        return

    # Preprocess the data
    print(f"Reading and preprocessing {excel_file}...")
    processed_df = preprocess_gallstone_data(excel_file)

    # Display basic statistics
    print(f"\nDataset shape: {processed_df.shape}")
    print(f"Number of instances: {len(processed_df)}")
    print(f"Number of attributes: {len(processed_df.columns)}")

    print("\nAttribute summary:")
    for col in processed_df.columns:
        print(f"  {col}: {processed_df[col].value_counts().to_dict()}")

    # Display ARFF header for Question 4a
    print("\n")
    display_arff_header()

    # Generate ARFF file
    output_file = 'gallstone.arff'
    generate_arff_file(processed_df, output_file)

    print(f"\nProcessing complete!")
    print(f"You can now load '{output_file}' into Weka for Question 4b.")
    print("\nFor Question 4a: The ARFF header (shown above) contains all sections")
    print("before the '@DATA' section.")


if __name__ == "__main__":
    main()
