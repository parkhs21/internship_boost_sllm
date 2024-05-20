function after_submit_clicked() {
    var output_box = document.getElementById("output_box");
    var output_box_wrap = output_box.getElementsByClassName("wrap")[0];
    var output_box_meta_text = output_box.getElementsByClassName("meta-text")[0];
    output_box_wrap.classList.add('translucent');
    output_box_wrap.classList.remove('hide');

    var time_hidden_btn = document.getElementById('hidden_btn');

    output_box_meta_text.textContent = time_hidden_btn.textContent + " s";
}