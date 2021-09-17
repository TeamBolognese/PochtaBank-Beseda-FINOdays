
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: extenstions ]
    import { createEventDispatcher } from 'svelte';
    const dispatch = createEventDispatcher();


	// [ IMPORTS: components ]
    
	
	// [ IMPORTS: props ]
	export let placeholder = 'Введите текст ...';


    // [ PROPS ]
    let value = '';


    // [ EVENTS ]
    $: togglePlaceholder( value );
    $: dispatch('TransmitMessage', value);


    // [ METHODS ]
    const togglePlaceholder = async ( value ) => {
        if( value ) {
            document.querySelector('p.placeholder').style.display = 'none';
        } else {
            document.querySelector('p.placeholder').style.display = 'block';
        }
    }

</script>


<!-- [ TEMPLATE: Input > Chat ] -->
<div class = 'input'>
    <div contenteditable = 'true' bind:textContent={ value }></div>
    <p class="placeholder">{ placeholder }</p>
</div>
