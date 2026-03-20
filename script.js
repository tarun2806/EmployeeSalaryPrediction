let importanceChart;

document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Collect data
    const data = {
        Age: parseInt(document.getElementById('Age').value),
        Gender: document.getElementById('Gender').value,
        Department: document.getElementById('Department').value,
        Job_Title: document.getElementById('Job_Title').value,
        Experience: parseInt(document.getElementById('Experience').value),
        Promotion: document.getElementById('Promotion').value,
        Satisfaction: parseFloat(document.getElementById('Satisfaction').value),
        Hours: parseInt(document.getElementById('Hours').value)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        
        if (result.error) {
            alert("Error: " + result.error);
            return;
        }

        // Show result section
        const resultSection = document.getElementById('resultSection');
        resultSection.classList.remove('hidden');
        resultSection.classList.add('animate-fadeIn');

        // Update UI
        const riskVal = (result.risk * 100).toFixed(1);
        document.getElementById('riskText').innerText = riskVal + "%";
        
        const riskBadge = document.getElementById('riskBadge');
        if (result.risk > 0.5) {
            riskBadge.innerText = "🔴 High Risk";
            riskBadge.style.color = "#ef4444";
            document.getElementById('insightText').innerText = `Significant attrition risk detected for this profile. Possible causes include low satisfaction or role mismatch. Immediate engagement is recommended.`;
        } else {
            riskBadge.innerText = "🟢 Safe";
            riskBadge.style.color = "#22c55e";
             document.getElementById('insightText').innerText = `This employee profile is highly stable with low turnover probability. Reward loyalty and maintain current engagement strategies.`;
        }

        document.getElementById('salText').innerText = "₹ " + Number(result.salary).toLocaleString();

        // Update Chart
        const ctx = document.getElementById('importanceChart').getContext('2d');
        
        if (importanceChart) {
            importanceChart.destroy();
        }

        importanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: result.features,
                datasets: [{
                    label: 'Impact Magnitude',
                    data: result.feat_imp,
                    backgroundColor: 'rgba(96, 165, 250, 0.6)',
                    borderColor: '#60a5fa',
                    borderWidth: 1,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                     legend: { display: false }
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                }
            }
        });

        // Smooth scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        console.error(err);
        alert("Failed to communicate with API. Please ensure Vercel backend is running.");
    }
});
