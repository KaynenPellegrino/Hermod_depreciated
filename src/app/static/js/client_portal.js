// src/app/static/js/client_portal.js

document.addEventListener('DOMContentLoaded', () => {
    loadProjects();
    setupCreateProjectForm();
});

/**
 * Fetches the list of projects from the server and displays them.
 */
function loadProjects() {
    fetch('/api/projects')
        .then(response => response.json())
        .then(data => {
            const projectList = document.getElementById('projectList');
            projectList.innerHTML = '';

            data.projects.forEach(project => {
                const li = document.createElement('li');
                li.className = 'project-item';

                const projectInfo = document.createElement('div');
                projectInfo.innerHTML = `
                    <h2>${project.project_name}</h2>
                    <p>${project.description}</p>
                `;

                const projectStatus = document.createElement('div');
                projectStatus.className = `status ${project.status.replace(' ', '-').toLowerCase()}`;
                projectStatus.textContent = project.status.replace('_', ' ').toUpperCase();

                li.appendChild(projectInfo);
                li.appendChild(projectStatus);

                projectList.appendChild(li);
            });
        })
        .catch(error => {
            console.error('Error fetching projects:', error);
            displayAlert('error', 'Failed to load projects. Please try again later.');
        });
}

/**
 * Sets up the event listener for the create project form.
 */
function setupCreateProjectForm() {
    const form = document.getElementById('createProjectForm');
    form.addEventListener('submit', event => {
        event.preventDefault();
        const formData = new FormData(form);
        const projectData = Object.fromEntries(formData.entries());

        fetch('/api/projects', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(projectData)
        })
            .then(response => response.json())
            .then(data => {
                if (data.project_id) {
                    displayAlert('success', 'Project created successfully!');
                    closeCreateProjectModal();
                    loadProjects();
                    form.reset();
                } else {
                    displayAlert('error', data.error || 'Failed to create project.');
                }
            })
            .catch(error => {
                console.error('Error creating project:', error);
                displayAlert('error', 'An error occurred while creating the project.');
            });
    });
}

/**
 * Opens the create project modal.
 */
function openCreateProjectModal() {
    const modal = document.getElementById('createProjectModal');
    modal.style.display = 'block';
}

/**
 * Closes the create project modal.
 */
function closeCreateProjectModal() {
    const modal = document.getElementById('createProjectModal');
    modal.style.display = 'none';
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
    const container = document.querySelector('.client-portal-container');
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    container.prepend(alertDiv);

    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}
