function getRecommendation() {
    const activity = document.getElementById('activitySelect').value;

    fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ activity: activity })
    })
    .then(response => response.json())
    .then(data => {
        const list = document.getElementById('recommendationsList');
        list.innerHTML = '';  // Clear previous recommendations
        if (data.length > 0) {
            data.forEach(track => {
                const li = document.createElement('li');
                li.textContent = `${track.track_name} by ${track.artists}`;
                list.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No recommendations available for this activity.';
            list.appendChild(li);
        }
    })
    .catch(error => console.error('Error:', error));
}
