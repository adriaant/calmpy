$(document).ready(function() {

    $(".confirm").confirm({
        text: "Are you sure you want to delete this network?",
        title: "Confirmation required",
        confirm: function(button) {
            var tr = button.closest('tr');
            var url = button.attr('href')
            $.ajax({
                url: url,
                type: 'DELETE',
                success: function(result) {
                    tr.css("background-color","#FF6666");
                    tr.fadeOut(400, function(){
                        tr.remove();
                    });
                }
            });
        },
        cancel: function(button) {
            // nothing to do
        },
        confirmButton: "Yes",
        cancelButton: "No",
        post: true,
        confirmButtonClass: "btn-danger",
        cancelButtonClass: "btn-default",
    });
});
