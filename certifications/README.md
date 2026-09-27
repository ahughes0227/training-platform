# Certified trainer runtimes

`defect runtime certify CONFIG --output certifications/RUNTIME_ID.yaml` writes one small record only after local trainer validation, GPU container validation, registry push, and Vertex GPU/GCS handshake pass for the same image digest. Commit the record so experiments can reuse it. The full image and DINOv3 weights live outside Git.
