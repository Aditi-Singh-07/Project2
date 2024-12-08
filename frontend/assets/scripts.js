/*document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('resumeInput');
    const skillsDiv = document.getElementById('skills');
    const errorDiv = document.getElementById('error');

    if (!fileInput.files.length) {
        errorDiv.innerHTML = '<p>Please upload a resume.</p>';
        return;
    }

    const formData = new FormData();
    formData.append('pdf', fileInput.files[0]);

    try {
        const response = await fetch('http://localhost:5000/upload', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        if (data.skills) {
            skillsDiv.innerHTML = `<pre>${JSON.stringify(data.skills, null, 2)}</pre>`;
            skillsDiv.innerHTML += '<a href="mcq.html"><button>Generate MCQs</button></a>';
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    } catch (error) {
        window.location.href = "error.html";
    }
});*/
