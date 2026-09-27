import numpy as np
import cv2
import os

"""
LSB embedding method.
Params: cover image and bits per pixel @ 0.4
Return: Stego image w/ lsb embedding
"""
def embed_lsb_matching(coverImg: np.ndarray, bpp=0.4):
    stego = coverImg.copy().astype(np.int16)
    flat = stego.flatten() # into 1D
    payloadSize = int(len(flat) * bpp)
    payload = np.random.randint(0,2,payloadSize)

# embed lsb
    for i, bit in enumerate(payload):
        if (flat[i] % 2) != bit:
            direction = np.random.choice([-1,1]) # lsb matching - random direction for + / -
            flat[i] = np.clip(flat[i]+direction,0,255) # perfom the modification

    # from flatten back to org
    # convert back to org data type
    return flat.reshape(stego.shape).astype(np.uint8)


"""
Embed cover imgs and save the stegos to stego_lsb folder.
Params: cover folder, lsb stego folder bits per pixel @ 0.4
Return: # of successfully processed images
"""

# lsb embeded stegp dataset generation method
def generate_stego_dataset(coverFolder, stegoFolder, bpp=0.4):

    # Create output folder if it does not exist
    os.makedirs(stegoFolder, exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0

    # Get valid image files and sort them
    files = sorted(
        [
            fname for fname in os.listdir(coverFolder)
            if fname.lower().endswith(
                ('.pgm', '.jpg', '.jpeg', '.png', '.bmp')
            )
        ],
        key=lambda x: int(os.path.splitext(x)[0])
        if os.path.splitext(x)[0].isdigit()
        else x
    )

    total = len(files)

    print(f"\nFound {total} images in {coverFolder}")
    print(f"Embedding rate: {bpp} bpp")
    print(f"Output folder: {stegoFolder}")
    print("\nStarting LSB embedding...\n")

    # Process images
    for index, fname in enumerate(files, start=1):

        coverPath = os.path.join(coverFolder, fname)
        outputPath = os.path.join(stegoFolder, fname)

        # Skip files that already exist
        if os.path.exists(outputPath):
            skipped += 1
            print(
                f"[{index}/{total}] {fname} - SKIPPED (already exists)",
                flush=True
            )
            continue

        try:

            # Load image
            img = cv2.imread(
                coverPath,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                print(
                    f"[{index}/{total}] {fname} - ERROR (could not load)",
                    flush=True
                )
                errors += 1
                continue

            # Perform LSB matching
            stego = embed_lsb_matching(img, bpp)

            # Save stego image
            success = cv2.imwrite(outputPath, stego)

            if not success:
                print(
                    f"[{index}/{total}] {fname} - ERROR (could not save)",
                    flush=True
                )
                errors += 1
                continue

            processed += 1

            print(
                f"[{index}/{total}] {fname} - DONE",
                flush=True
            )

        except Exception as e:

            errors += 1

            print(
                f"[{index}/{total}] {fname} - ERROR: {e}",
                flush=True
            )

    print("\n===================================")
    print("LSB EMBEDDING COMPLETE")
    print("===================================")
    print(f"New images generated : {processed}")
    print(f"Already existed      : {skipped}")
    print(f"Errors               : {errors}")
    print(f"Total                 : {total}")

    return processed

if __name__ == '__main__':
    generate_stego_dataset(coverFolder='data/cover', stegoFolder='data/stego_lsb', bpp=0.4)