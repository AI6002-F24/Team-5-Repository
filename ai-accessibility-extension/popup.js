// Function to get the active tab ID
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
  
  // Function to describe images on the page
  function describeImages(tabId) {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: function() {
        // Function to describe images on the page
        const images = document.querySelectorAll("img");
        images.forEach((img) => {
          // Replace with actual image description logic
          console.log("Found image:", img.src);
          // Example: Use a library like TensorFlow.js to analyze the image
          // const imgTensor = tf.browser.fromPixels(img);
          // const predictions = await model.predict(imgTensor);
          // console.log("Image description:", predictions);
        });
      },
    }, (result) => {
      if (chrome.runtime.lastError) {
        console.error("Error executing script:", JSON.stringify(chrome.runtime.lastError, null, 2));
      }
    });
  }
  
  // Function to perform OCR on images
  function performOcr(tabId) {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: function() {
        // Function to perform OCR on images
        const images = document.querySelectorAll("img");
        images.forEach((img) => {
          // Replace with actual OCR logic
          console.log("Running OCR on image:", img.src);
          // Example: Use a library like Tesseract.js to perform OCR
          // const ocrResult = await Tesseract.recognize(img.src);
          // console.log("OCR result:", ocrResult);
        });
      },
    }, (result) => {
      if (chrome.runtime.lastError) {
        console.error("Error executing script:", chrome.runtime.lastError.message, chrome.runtime.lastError.stack);
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
  
  // Event listener for "Read Text" button
  document.getElementById("ocrBtn").addEventListener("click", () => {
    console.log("Read Text button clicked.");
    getActiveTabId((tabId) => {
      if (tabId) {
        performOcr(tabId);
      }
    });
  });