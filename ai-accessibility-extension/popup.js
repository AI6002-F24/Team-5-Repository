function getActiveTabId(callback) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs.length > 0) {
      callback(tabs[0].id);
    } else {
      console.error("No active tab found.");
      callback(null);
    }
  });
}

function describeImages(tabId) {
    chrome.scripting.executeScript({
        target: { tabId: tabId },
        func: async function() {
            async function getImageBase64FromImg(img) {
                return new Promise((resolve, reject) => {
                    try {
                        const newImg = new Image();
                        newImg.crossOrigin = 'Anonymous';
                        newImg.src = img.src;

                        newImg.onload = function() {
                            const canvas = document.createElement('canvas');
                            canvas.width = newImg.naturalWidth;
                            canvas.height = newImg.naturalHeight;

                            const ctx = canvas.getContext('2d');
                            ctx.drawImage(newImg, 0, 0);

                            const dataUrl = canvas.toDataURL('image/jpeg');
                            resolve(dataUrl);
                        };

                        newImg.onerror = function(error) {
                            reject('Failed to load image: ' + error);
                        };
                    } catch (error) {
                        reject(error);
                    }
                });
            }

            const images = document.querySelectorAll("img");
            console.log(`Found ${images.length} images on the page`);

            for (const img of images) {
                try {
                    const rect = img.getBoundingClientRect();
                    if (rect.width < 50 || rect.height < 50) {
                        console.log("Skipping small image:", img.src);
                        continue;
                    }

                    console.log("Processing image:", img.src);

                    try {
                        const imageBase64 = await getImageBase64FromImg(img);

                        console.log("Sending request to model...");

                        const response = await fetch('https://6438-34-142-132-1.ngrok-free.app/generate-caption', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                image: imageBase64
                            })
                        });

                        console.log("Response status:", response.status);

                        const data = await response.json();

                        if (data.status === "success" && data.caption) {
                            console.log("🎯 Caption generated successfully:", data.caption);
                            img.title = data.caption;

                            const captionDiv = document.createElement('div');
                            captionDiv.style.maxWidth = img.width + 'px';
                            captionDiv.style.padding = '8px';
                            captionDiv.style.backgroundColor = '#f0f0f0';
                            captionDiv.style.margin = '4px 0';
                            captionDiv.style.borderRadius = '4px';
                            captionDiv.textContent = `🖼️ ${data.caption}`;
                            img.insertAdjacentElement('afterend', captionDiv);
                        } else {
                            console.error("Error in response:", data.error || "Unknown error");
                        }
                    } catch (error) {
                        console.error('Error processing image:', error);
                    }
                } catch (error) {
                    console.error('Error with image:', img.src, error);
                }
            }
        }
    }, (result) => {
        if (chrome.runtime.lastError) {
            console.error("Error executing script:", JSON.stringify(chrome.runtime.lastError, null, 2));
        } else {
            console.log("Script executed successfully");
        }
    });
}

// Event listener for "Describe Image" button
document.getElementById("describeBtn").addEventListener("click", () => {
  console.log("Describe Image button clicked.");
  getActiveTabId((tabId) => {
    if (tabId) {
      describeImages(tabId);
    }
  });
});
