/**
 * Plant Nutrient Detective
 * Main JavaScript file for enhancing the user experience
 */

document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Image preview for file upload if on detect page
    const fileInput = document.getElementById('file');
    if (fileInput) {
        const previewContainer = document.createElement('div');
        previewContainer.className = 'image-preview mt-3 d-none';
        previewContainer.innerHTML = '<img id="image-preview" class="img-fluid rounded" alt="Image Preview">';
        
        fileInput.parentNode.insertAdjacentElement('afterend', previewContainer);
        
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                
                reader.addEventListener('load', function() {
                    previewContainer.classList.remove('d-none');
                    document.getElementById('image-preview').src = reader.result;
                });
                
                reader.readAsDataURL(file);
            } else {
                previewContainer.classList.add('d-none');
            }
            
            // Update file name display
            if (file) {
                const fileName = file.name;
                const nextSibling = this.nextElementSibling;
                if (nextSibling) {
                    nextSibling.innerText = fileName;
                }
            }
        });
    }
    
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggerList.length > 0 && typeof bootstrap !== 'undefined') {
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
    
    // Animation on scroll for elements with .animate-on-scroll class
    function animateOnScroll() {
        const elements = document.querySelectorAll('.animate-on-scroll');
        
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            
            if (elementPosition < screenPosition) {
                element.classList.add('animate__animated', element.dataset.animation || 'animate__fadeIn');
            }
        });
    }
    
    // Run animation check on load and scroll
    if (document.querySelectorAll('.animate-on-scroll').length > 0) {
        window.addEventListener('scroll', animateOnScroll);
        animateOnScroll(); // Initial check
    }
    
    // Plant type selection highlight
    const plantTypeInputs = document.querySelectorAll('input[name="plant_type"]');
    if (plantTypeInputs.length > 0) {
        plantTypeInputs.forEach(input => {
            input.addEventListener('change', function() {
                const allCards = document.querySelectorAll('.plant-type-card .card');
                allCards.forEach(card => {
                    card.classList.remove('border-success');
                    card.classList.add('border-light');
                });
                
                if (this.checked) {
                    const selectedCard = this.nextElementSibling.querySelector('.card');
                    selectedCard.classList.remove('border-light');
                    selectedCard.classList.add('border-success');
                }
            });
        });
        
        // Set initial selection
        const checkedInput = document.querySelector('input[name="plant_type"]:checked');
        if (checkedInput) {
            const selectedCard = checkedInput.nextElementSibling.querySelector('.card');
            selectedCard.classList.remove('border-light');
            selectedCard.classList.add('border-success');
        }
    }

    // Set width for probability bars
    document.querySelectorAll('.prob-bar').forEach(function(bar) {
        const probability = bar.getAttribute('data-probability');
        bar.style.width = probability + '%';
    });

    // Set width for progress bars
    document.querySelectorAll('.progress-bar').forEach(function(progressBar) {
        const value = progressBar.getAttribute('aria-valuenow');
        progressBar.style.width = value + '%';
    });

    // Form validation for detect page
    const forms = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}); 