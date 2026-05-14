import { useState } from "react";

function Upload() {

  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleUpload = async () => {

    if (!file) {

      setMessage("Please select a file");
      return;
    }

    const formData = new FormData();

    formData.append("file", file);

    try {

      const response = await fetch(
        "http://127.0.0.1:8000/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (data.success) {

        setMessage(
          `File uploaded successfully | Stored ${data.chunks_stored} chunks`
        );

      } else {

        setMessage(data.message);
      }

    } catch (error) {

      console.error(error);

      setMessage("Upload failed");
    }
  };

  return (
    <div>

      <h2>Upload Documents</h2>

      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button
        onClick={handleUpload}
        style={{
          marginLeft: "10px",
          padding: "10px"
        }}
      >
        Upload
      </button>

      <p>{message}</p>

    </div>
  );
}

export default Upload;