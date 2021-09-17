
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: components ]
    
	
	// [ IMPORTS: props ]
	export let type;


</script>


<!-- [ TEMPLATE: Ico > Chat ] -->
{#if type == 'send'}
    <svg class="ico">
        <use xlink:href="./img/sprites.svg#ico-send"></use>
    </svg>
{/if}