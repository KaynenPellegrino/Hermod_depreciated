// src/app/static/js/scripts.js

/**
 * Logs out the user by redirecting to the logout endpoint.
 */
function logout() {
    if (confirm('Are you sure you want to logout?')) {
        window.location.href = '/logout';
    }
}

/**
 * Initializes modal functionalities.
 */
document.addEventListener('DOMContentLoaded', () => {
    setupModalClose();
});

/**
 * Sets up event listeners to close modals when clicking outside or on the close button.
 */
function setupModalClose() {
    // Close modal when clicking on the close button
    const closeButtons = document.querySelectorAll('.close-modal');
    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            const modal = button.closest('.modal');
            if (modal) {
                modal.style.display = 'none';
            }
        });
    });

    // Close modal when clicking outside the modal content
    window.addEventListener('click', event => {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        });
    });
}

/**
 * Displays an alert message.
 * @param {string} type - The type of alert ('success' or 'error').
 * @param {string} message - The message to display.
 */
function displayAlert(type, message) {
    const container = document.querySelector('.container') || document.body;
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    container.prepend(alertDiv);

    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}
