const form = document.getElementById('prediction-form');
const resultBox = document.getElementById('result');
const errorBox = document.getElementById('error');
const predictionLabel = document.getElementById('prediction-label');
const probabilitiesBox = document.getElementById('probabilities');

function hideAll() {
  resultBox.classList.add('hidden');
  errorBox.classList.add('hidden');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  hideAll();

  const payload = {
    sepal_length: Number(document.getElementById('sepal_length').value),
    sepal_width: Number(document.getElementById('sepal_width').value),
    petal_length: Number(document.getElementById('petal_length').value),
    petal_width: Number(document.getElementById('petal_width').value),
  };

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Prediction failed');
    }

    predictionLabel.textContent = `Predicted species: ${data.prediction}`;
    probabilitiesBox.innerHTML = '';

    Object.entries(data.probabilities).forEach(([label, probability]) => {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `<span>${label}</span><strong>${probability.toFixed(4)}</strong>`;
      probabilitiesBox.appendChild(row);
    });

    resultBox.classList.remove('hidden');
  } catch (error) {
    errorBox.textContent = `Error: ${error.message}`;
    errorBox.classList.remove('hidden');
  }
});
