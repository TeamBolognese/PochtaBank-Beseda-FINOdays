
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: extensions ]


	// [ IMPORTS: components ]

	
	// [ IMPORTS: props ]
    export let message = 'message';
    export let date = '00:00';
    export let isactive = '';
    export let emotion = 'positive';

    
	// [ PROPS ]
	



</script>


<!-- [ TEMPLATE: Message > MessageContainer ] -->
<div class = 'message { isactive }'>
    
    <!-- message text -->
    <div class="message-text">
        { message }
    </div>
    
    <!-- meta data -->
	<div class="metadata">
        <div class="indicator">
            {#if emotion == 'positive'}
                <span class = 'emotion green'>положительное</span>
            {:else if emotion == 'neutral'}
                <span class = 'emotion yellow'>нейтральное</span>
            {:else}
                <span class = 'emotion red'>отрицательное</span>
            {/if}
        </div>
        <div class="date">{ date }</div>
    </div>

</div>
