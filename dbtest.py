from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["E_Commerce"]
collection = db["samples"]

# This will create db + collection automatically if not exist
collection.insert_one({"test": "works!"})

print(list(collection.find()))

def populate_database(json_file="E_Commerce.samples.json"):
    mongo_collection, chroma_collection, dataset_folder = initialize_databases()

    with open(json_file, "r") as f:
        products = json.load(f)

    for product in products:
        product_id = str(product["_id"]["$oid"])  # extract MongoDB _id
        image_url = product.get("image")
        image_path = os.path.join(dataset_folder, f"{product_id}.jpg")

        # 1. Download image
        if image_url:
            try:
                resp = requests.get(image_url, timeout=10)
                if resp.status_code == 200:
                    with open(image_path, "wb") as img_file:
                        img_file.write(resp.content)
                else:
                    print(f"⚠️ Failed to download {image_url}")
            except Exception as e:
                print(f"⚠️ Error downloading {image_url}: {e}")

        # 2. Insert into MongoDB
        try:
            mongo_collection.insert_one(product)
        except Exception:
            # Ignore if already exists
            pass

        # 3. Add to ChromaDB
        try:
            chroma_collection.add(
                ids=[product_id],
                documents=[product.get("description", "")],
                metadatas=[{
                    "title": product.get("title"),
                    "category": product.get("category"),
                    "price": product.get("price"),
                    "image_path": image_path
                }],
            )
        except Exception as e:
            print(f"⚠️ Error adding to Chroma: {e}")

    print("✅ Database repopulated successfully!")