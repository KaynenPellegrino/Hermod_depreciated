// static/js/scripts.js

// Establish a connection to the SocketIO server
const socket = io();

// Listen for the 'dashboard_update' event
socket.on('dashboard_update', (data) => {
    console.log('Received dashboard update:', data);
    updateDashboard(data);
});

function updateDashboard(data) {
    // Update title
    document.getElementById('dashboard-title').innerText = data.title;

    // Update metrics
    const metricsList = document.getElementById('metrics-list');
    metricsList.innerHTML = '';
    for (const [metric, value] of Object.entries(data.metrics)) {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${metric.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}:</strong> <span>${value}</span>`;
        metricsList.appendChild(li);
    }

    // Update alerts
    const alertsList = document.getElementById('alerts-list');
    alertsList.innerHTML = '';
    data.alerts.forEach(alert => {
        const li = document.createElement('li');
        li.className = alert.type;
        li.innerText = alert.message;
        alertsList.appendChild(li);
    });

    // Update tools (if necessary)
    // If the list of tools can change, implement similar logic
}
