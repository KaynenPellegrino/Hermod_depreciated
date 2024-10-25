// src/app/static/js/main.js

/**
 * Logs out the user by redirecting to the logout endpoint.
 */
function logout() {
    if (confirm('Are you sure you want to logout?')) {
        window.location.href = '/logout';
    }
}

/**
 * Initializes global functionalities.
 */
document.addEventListener('DOMContentLoaded', () => {
    setupNavigationLinks();
});

/**
 * Sets up event listeners for navigation links.
 */
function setupNavigationLinks() {
    const links = document.querySelectorAll('.navbar a');
    links.forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            const url = link.getAttribute('href');
            navigateTo(url);
        });
    });
}

/**
 * Navigates to a specified URL using fetch and updates the content dynamically.
 * @param {string} url - The URL to navigate to.
 */
function navigateTo(url) {
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok.');
            }
            return response.text();
        })
        .then(html => {
            document.body.innerHTML = html;
            // Reinitialize scripts after navigation
            const scriptTags = document.querySelectorAll('script');
            scriptTags.forEach(script => {
                const newScript = document.createElement('script');
                if (script.src) {
                    newScript.src = script.src;
                } else {
                    newScript.textContent = script.textContent;
                }
                document.body.appendChild(newScript);
            });
        })
        .catch(error => {
            console.error('Error navigating:', error);
            alert('Failed to load the requested page.');
        });
}
