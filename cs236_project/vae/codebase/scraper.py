import os
import requests

def download_images(api_key, query, output_folder, count=10):
    endpoint = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": count}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses.
        data = response.json()

        if "value" in data:
            for i, item in enumerate(data["value"]):
                image_url = item["contentUrl"]
                try:
                    image_response = requests.get(image_url)
                    image_response.raise_for_status()  # Raise an HTTPError for bad responses.

                    # Check if the content type is an image (JPEG, PNG, etc.) before saving.
                    content_type = image_response.headers.get("Content-Type", "").lower()
                    if content_type.startswith("image"):
                        with open(os.path.join(output_folder, f"{query}_{i+1}.jpg"), "wb") as f:
                            f.write(image_response.content)
                    else:
                        print(f"Skipping non-image content at URL: {image_url}")

                except requests.exceptions.HTTPError as img_err:
                    print(f"Error downloading image: {img_err}")

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Request Exception: {err}")

if __name__ == "__main__":
    api_key = "35df69fc73f34d809457384b04d7f0fc"
    query = "Kirby"
    output_folder = "kirby_images"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    download_images(api_key, query, output_folder, count=5000)
