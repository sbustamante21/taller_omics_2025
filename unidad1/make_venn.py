import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import sys

def read_file(file_path):
    """Reads a file and returns a set of elements (one per line)."""
    with open(file_path, 'r') as file:
        return set(line.strip() for line in file)

def write_set_to_file(filename, elements):
    """Writes a set of elements to a file, one per line."""
    with open(filename, 'w') as file:
        for element in sorted(elements):
            file.write(f"{element}\n")

def create_venn_diagram(files, output_image):
    """Creates a Venn diagram for up to 4 sets and saves unique/intersection elements."""
    sets = [read_file(f) for f in files]
    num_sets = len(sets)

    plt.figure(figsize=(6, 6))

    if num_sets == 2:
        venn = venn2([sets[0], sets[1]], set_labels=(f"Set 1 ({files[0]})", f"Set 2 ({files[1]})"))

        # Compute unique & intersection sets
        unique_set1 = sets[0] - sets[1]
        unique_set2 = sets[1] - sets[0]
        intersection = sets[0] & sets[1]

        # Save results
        write_set_to_file("unique_set1.txt", unique_set1)
        write_set_to_file("unique_set2.txt", unique_set2)
        write_set_to_file("intersection.txt", intersection)

    elif num_sets == 3:
        venn = venn3([sets[0], sets[1], sets[2]], set_labels=(f"Set 1 ({files[0]})", f"Set 2 ({files[1]})", f"Set 3 ({files[2]})"))

        # Compute unique & intersection sets
        unique_set1 = sets[0] - (sets[1] | sets[2])
        unique_set2 = sets[1] - (sets[0] | sets[2])
        unique_set3 = sets[2] - (sets[0] | sets[1])
        intersection = sets[0] & sets[1] & sets[2]

        # Save results
        write_set_to_file("unique_set1.txt", unique_set1)
        write_set_to_file("unique_set2.txt", unique_set2)
        write_set_to_file("unique_set3.txt", unique_set3)
        write_set_to_file("intersection.txt", intersection)

    elif num_sets == 4:
        print("Venn diagrams for 4 sets are not well-supported. Showing pairwise overlaps instead.")
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Creating pairwise Venn diagrams
        plt.sca(axes[0, 0])
        venn2([sets[0], sets[1]], set_labels=(f"Set 1 ({files[0]})", f"Set 2 ({files[1]})"))
        
        plt.sca(axes[0, 1])
        venn2([sets[0], sets[2]], set_labels=(f"Set 1 ({files[0]})", f"Set 3 ({files[2]})"))
        
        plt.sca(axes[1, 0])
        venn2([sets[0], sets[3]], set_labels=(f"Set 1 ({files[0]})", f"Set 4 ({files[3]})"))
        
        plt.sca(axes[1, 1])
        venn2([sets[1], sets[2]], set_labels=(f"Set 2 ({files[1]})", f"Set 3 ({files[2]})"))

        plt.tight_layout()

    plt.title("Venn Diagram")
    plt.savefig(output_image)
    plt.show()

# Usage example
if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print("Usage: python venn_diagram.py <file1> <file2> [<file3>] [<file4>] <output_image>")
        sys.exit(1)

    input_files = sys.argv[1:-1]  # All input files except the last argument (output image)
    output_image = sys.argv[-1]

    create_venn_diagram(input_files, output_image)

