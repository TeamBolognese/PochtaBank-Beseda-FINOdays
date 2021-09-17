
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: components ]
	import SingleAction from "./SingleAction.svelte";
	
	// [ IMPORTS: props ]
	export let actions = [];

</script>


<!-- [ TEMPLATE: Actions > Analytics > Dashboard ] -->
<div class = 'section-actions'>

	<h2 class = 'section-title'>Возможные действия:</h2>

    <div class="content">
        {#each actions as action}
            <SingleAction { action } />
        {/each}
    </div>

</div>