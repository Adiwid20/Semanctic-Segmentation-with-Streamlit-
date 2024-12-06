

// Switch between light and dark mode
document.addEventListener('DOMContentLoaded', function () {
    const toggleModeButton = document.querySelector('#toggle-mode');

    if (toggleModeButton) {
        toggleModeButton.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            document.body.classList.toggle('light-mode');
        });
    }
});