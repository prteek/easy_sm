// Enhanced copy button feedback
document.addEventListener('DOMContentLoaded', function() {
  // Add copy feedback for code blocks
  document.querySelectorAll('.md-clipboard').forEach(function(button) {
    button.addEventListener('click', function() {
      button.classList.add('copied');
      setTimeout(function() {
        button.classList.remove('copied');
      }, 2000);
    });
  });
});
