def individual_serial(image) -> dict:
    return {
        "id": str(image["_id"]),
        "path": image["path"],
        "value": image["value"],
    }
def list_serial(images) -> list:
    return [individual_serial(image) for image in images]