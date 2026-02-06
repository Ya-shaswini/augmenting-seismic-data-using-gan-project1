import os

class KNetParser:
    def parse_file(self, filepath):
        """
        Parses a K-NET ASCII format file and returns acceleration data in Gal.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Extract Scale Factor
        scale_factor = 1.0
        data_start_line = 0
        
        for i, line in enumerate(lines):
            if "Scale Factor" in line:
                try:
                    parts = line.split()
                    # format: Scale Factor      numerator(gal)/denominator
                    # parts might be ['Scale', 'Factor', '7845(gal)/8223790']
                    raw_factor = parts[-1]
                    num_str, den_str = raw_factor.split("(gal)/")
                    numerator = float(num_str)
                    denominator = float(den_str)
                    scale_factor = numerator / denominator
                except Exception as e:
                    print(f"Warning: Could not parse scale factor, using 1.0. Error: {e}")
            
            if "Memo." in line:
                data_start_line = i + 1
                break
        
        if data_start_line == 0:
            raise ValueError("Could not find start of data (Memo. tag missing)")

        # Parse Data
        data_values = []
        for line in lines[data_start_line:]:
            # line contains space separated integers
            values = line.strip().split()
            for v in values:
                try:
                    data_values.append(float(v) * scale_factor)
                except ValueError:
                    continue

        return data_values
