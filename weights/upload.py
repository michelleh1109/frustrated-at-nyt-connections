import modal

app = modal.App("upload-weights")
volume = modal.Volume.from_name("lora-weights", create_if_missing=True)

@app.local_entrypoint()
def main(local_path: str, remote_path: str):
    with volume.batch_upload() as upload:
        upload.put_directory(local_path, remote_path)