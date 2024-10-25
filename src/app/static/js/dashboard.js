// src/app/static/js/dashboard.js

document.addEventListener('DOMContentLoaded', () => {
    loadStatistics();
    loadRecentActivities();
});

/**
 * Fetches and displays dashboard statistics.
 */
function loadStatistics() {
    fetch('/api/dashboard/stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('totalProjects').textContent = data.total_projects;
            document.getElementById('activeUsers').textContent = data.active_users;
            document.getElementById('completedTasks').textContent = data.completed_tasks;
        })
        .catch(error => {
            console.error('Error fetching statistics:', error);
            displayAlert('error', 'Failed to load statistics.');
        });
}

/**
 * Fetches and displays recent activities.
 */
function loadRecentActivities() {
    fetch('/api/dashboard/activities')
        .then(response => response.json())
        .then(data => {
            const activityList = document.getElementById('activityList');
            activityList.innerHTML = '';

            data.activities.forEach(activity => {
                const li = document.createElement('li');
                li.className = 'activity-item';

                li.innerHTML = `
                    <span class="description">${activity.description}</span>
                    <span class="timestamp">${new Date(activity.timestamp).toLocaleString()}</span>
                `;

                activityList.appendChild(li);
            });
        })
        .catch(error => {
            console.error('Error fetching activities:', error);
            displayAlert('error', 'Failed to load recent activities.');
        });
}

/**
 * Logs out the user by redirecting to the logout endpoint.
 */
function logout() {
    window.location.href = '/logout';
}

/**
 * Displays an alert message.
 * @param {string} type - The type of alert ('success' or 'error').
 * @param {string} message - The message to display.
 */
function displayAlert(type, message) {
    const container = document.querySelector('.dashboard-container');
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    container.prepend(alertDiv);

    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}
