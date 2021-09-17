
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: components ]
    
	
	// [ IMPORTS: props ]
	export let type;


</script>


<!-- [ TEMPLATE: Emoji > Assessment ] -->
{#if type == 'positive'}
    <svg class="emoji">
        <use xlink:href="./img/sprites.svg#positive"></use>
    </svg>
{:else if type == 'neutral'}
    <svg class="emoji">
        <use xlink:href="./img/sprites.svg#neutral"></use>
    </svg>
{:else}
    <svg class="emoji">
        <use xlink:href="./img/sprites.svg#negative"></use>
    </svg>
{/if}